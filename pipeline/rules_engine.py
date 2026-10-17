"""Правила: факты по окнам (от VLM) -> события-нарушения. Чистый Python, без модели — можно быстро
перебирать правила на сохранённых фактах (scripts/rules_eval.py).

timeline: [{"lo": сек, "hi": сек, "f": {факты из pipeline.vlm.FACTS, "action": "..."}}, ...] по времени.
Каждое правило смотрит на текущее окно и историю. Включение/выключение: переменная RULES_OFF="id1,id2".
"""
import os

OFF = {x.strip() for x in os.getenv("RULES_OFF", "").split(",") if x.strip()}


def _visits(timeline):
    """Отрезки подряд идущих окон, где у стола есть посетитель."""
    cur = []
    for i, w in enumerate(timeline):
        if w["f"].get("visitor_count", 0) >= 1:
            cur.append(i)
        elif cur:
            yield cur
            cur = []
    if cur:
        yield cur


def apply(timeline: list[dict]) -> list[dict]:
    hits = []  # (window_index, type_id, why)
    cashier_seen = False
    for i, w in enumerate(timeline):
        f = w["f"]
        prev = timeline[i - 1]["f"] if i else {}
        cashier = f["cashier_at_desk"]
        others_on_cashier_side = f["people_count"] - f["visitor_count"] - (1 if cashier else 0)

        # 1. Ценности без присмотра: деньги/сумка на столе, кассир уже работал и ушёл с места
        if cashier_seen and not cashier and (f["money_on_desk"] or f["bag_on_desk"]):
            hits.append((i, "unattended_valuables",
                         "на столе остались " + ("деньги" if f["money_on_desk"] else "инкассаторская сумка")
                         + ", кассира на рабочем месте нет"))

        # 2. Рабочее место оставлено: кассир ушёл, ноутбук открыт / свет выключен
        if cashier_seen and not cashier and f["laptop_open"] and (
                f["people_count"] == 0 or f["lights_off"] or f["person_leaving"]):
            hits.append((i, "workplace_abandoned", "кассир покинул рабочее место, ноутбук остался открытым"))

        # 3. Сейф не защищён
        if f["keys_in_safe_lock"]:
            hits.append((i, "safe_unsecured", "ключи оставлены в замке сейфа"))
        elif f["safe_open"] and (not cashier or f["counting_machine_used"]):
            hits.append((i, "safe_unsecured",
                         "сейф открыт, пока кассир " + ("отсутствует" if not cashier else "работает с деньгами")))

        # 4. Посторонний: в кабинете кто-то есть, кассира на месте нет, человек входит или трогает технику;
        #    или лишний человек на стороне кассира
        if f["people_count"] >= 1 and not cashier and (
                f["touching_equipment"] or f["counting_machine_used"] or f["person_entering"]):
            hits.append((i, "unauthorized_person", "в кассовой зоне находится человек в отсутствие кассира"))
        elif cashier and others_on_cashier_side >= 1:
            hits.append((i, "unauthorized_person", "рядом с кассиром на его стороне стола находится посторонний"))

        # 5. Смешение операций: два посетителя сразу, или новый посетитель приходит, пока на столе деньги прежнего
        if f["visitor_count"] >= 2:
            hits.append((i, "mixed_operations", "у кассы одновременно несколько клиентов"))
        elif f["person_entering"] and f["money_on_desk"] and (prev.get("person_leaving") or f["person_leaving"]):
            hits.append((i, "mixed_operations",
                         "новый клиент подходит, пока на столе остаются деньги предыдущего"))

        # 6. Инкассация: сумку вскрывают в одиночку
        if f["bag_being_opened"] and f["people_count"] <= 1:
            hits.append((i, "collection_procedure", "инкассаторскую сумку вскрывают в одиночку"))

        cashier_seen = cashier_seen or cashier

    # 7. Порядок приёма наличных — по каждому визиту клиента
    for visit in _visits(timeline):
        F = [timeline[i]["f"] for i in visit]
        g = lambda f, k: bool(f.get(k))  # noqa: E731 — старые факты могут не иметь новых ключей
        took = [i for i in visit if g(timeline[i]["f"], "money_to_cashier") or g(timeline[i]["f"], "money_handed_over")]
        counted = [i for i in visit if g(timeline[i]["f"], "counting_machine_used")]
        # 7а. деньги приняты, документ за весь визит не предъявлялся
        if (took or counted) and not any(g(f, "id_document_visible") for f in F):
            hits.append(((took or counted)[0], "cash_intake_procedure",
                         "кассир принимает деньги, удостоверение личности клиента не проверялось"))
        # 7б. приняли деньги, а потом часть отдали обратно в руки клиенту в том же визите
        if took:
            back = [i for i in visit if i > took[0] and g(timeline[i]["f"], "money_to_visitor")]
            if back:
                hits.append((back[0], "cash_intake_procedure",
                             "после приёма и пересчёта часть денег возвращена клиенту в руки"))
        # 7в. клиент уходит, пока его деньги ещё пересчитываются
        for i in counted:
            f = timeline[i]["f"]
            nxt = timeline[i + 1]["f"] if i + 1 < len(timeline) else {}
            if g(f, "person_leaving") and (g(nxt, "counting_machine_used") or g(f, "money_on_desk")):
                hits.append((i, "cash_intake_procedure", "клиент ушёл, не дождавшись окончания пересчёта"))
                break

    hits = [h for h in hits if h[1] not in OFF]

    # склейка: подряд идущие окна одного типа -> одно событие
    events = []
    for i, tid, why in sorted(hits):
        last = next((e for e in reversed(events) if e["type_id"] == tid), None)
        if last and i <= last["last_i"] + 1:
            last["last_i"] = i
            last["n"] += 1
            continue
        events.append({"type_id": tid, "first_i": i, "last_i": i, "n": 1, "why": why})

    for e in events:
        w0, w1 = timeline[e["first_i"]], timeline[e["last_i"]]
        e["lo"], e["hi"] = w0["lo"], w1["hi"]
        e["score"] = min(0.95, 0.6 + 0.1 * e["n"])
        act = w0["f"].get("action", "")
        e["description"] = (f"{act.rstrip('.')}. " if act else "") + f"Признак нарушения: {e['why']}."
    return events
