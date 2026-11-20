// Shared behaviour for the design examples: themes, animated controls, the race
// pace card and a few content builders. Plain JavaScript, no build step.
(function () {
  "use strict";

  // ---------- Teams ----------
  // Brand-anchored colours tuned for separation on the chart canvas. Identity never
  // rests on colour alone: every bar carries its team name, and Haas (the second
  // red) also carries a stripe texture.
  const TEAMS = {
    mercedes: { name: "Mercedes", color: "#00A19C" },
    redbull: { name: "Red Bull", color: "#3A5BD9" },
    ferrari: { name: "Ferrari", color: "#F01C1C" },
    mclaren: { name: "McLaren", color: "#FF8000" },
    aston: { name: "Aston Martin", color: "#1E7F55" },
    alpine: { name: "Alpine", color: "#F25CA2" },
    williams: { name: "Williams", color: "#00A0DE" },
    rb: { name: "RB", color: "#7B61FF" },
    haas: { name: "Haas", color: "#C8102E", hatch: true },
    sauber: { name: "Sauber", color: "#3DBB3D" },
  };
  const ALIAS = {
    Mercedes: "mercedes", "Red Bull Racing": "redbull", "Red Bull": "redbull", Ferrari: "ferrari",
    McLaren: "mclaren", "Aston Martin": "aston", Alpine: "alpine", "Alpine F1 Team": "alpine",
    Williams: "williams", RB: "rb", "RB F1 Team": "rb", "Haas F1 Team": "haas",
    "Kick Sauber": "sauber", Sauber: "sauber",
  };
  function team(name) {
    const key = ALIAS[name];
    return key ? Object.assign({ key }, TEAMS[key]) : { key: name, name, color: "#888888" };
  }

  // ---------- Themes ----------
  const THEMES = [
    { id: "carbon", label: "Carbon", a: "#ff3b30", b: "#1d2027" },
    { id: "paddock", label: "Paddock", a: "#22d3ee", b: "#16223d" },
    { id: "heritage", label: "Heritage", a: "#d4a72c", b: "#1f6b45" },
    { id: "apex", label: "Apex", a: "#ffb020", b: "#7a35e8" },
  ];
  const root = document.documentElement;
  function store(key, value) {
    try { if (value === undefined) return localStorage.getItem(key); localStorage.setItem(key, value); } catch (e) { return null; }
    return null;
  }
  function applyTheme(theme, mode) {
    if (theme) { root.dataset.theme = theme; store("f1x-theme", theme); }
    if (mode) { root.dataset.mode = mode; store("f1x-mode", mode); }
    document.dispatchEvent(new CustomEvent("themechange"));
  }
  function initTheme() {
    const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    // ?theme=paddock&mode=light opens a given combination directly.
    const q = new URLSearchParams(location.search);
    const theme = THEMES.some((t) => t.id === q.get("theme")) ? q.get("theme") : null;
    const mode = ["dark", "light"].includes(q.get("mode")) ? q.get("mode") : null;
    applyTheme(theme || store("f1x-theme") || "carbon", mode || store("f1x-mode") || (dark ? "dark" : "light"));
    const ready = () => setTimeout(() => root.classList.add("theme-ready"), 60);
    if (document.readyState === "complete") ready(); else window.addEventListener("load", ready);
  }

  // Theme picker: four theme swatches and a dark/light switch.
  function mountThemePicker(el) {
    el.classList.add("theme-picker");
    const dots = THEMES.map((t) => {
      const b = document.createElement("button");
      b.className = "theme-dot"; b.type = "button"; b.title = t.label + " theme";
      b.setAttribute("aria-label", t.label + " theme");
      b.style.setProperty("--a", t.a); b.style.setProperty("--b", t.b);
      b.addEventListener("click", () => { applyTheme(t.id); sync(); });
      el.appendChild(b);
      return [t.id, b];
    });
    const mode = document.createElement("div");
    mode.innerHTML = '<button type="button" data-value="dark">Dark</button><button type="button" data-value="light">Light</button>';
    el.appendChild(mode);
    const control = segmented(mode, root.dataset.mode, (v) => applyTheme(null, v));
    function sync() { dots.forEach(([id, b]) => b.setAttribute("aria-pressed", String(id === root.dataset.theme))); control.set(root.dataset.mode, false); }
    sync();
  }

  // ---------- Animated controls ----------
  // Segmented control: buttons with data-value inside `el`; a thumb slides between
  // them with a spring.
  function segmented(el, value, onChange) {
    el.classList.add("seg");
    const thumb = document.createElement("span");
    thumb.className = "seg-thumb";
    el.prepend(thumb);
    const buttons = Array.from(el.querySelectorAll("button"));
    let current = value || buttons[0].dataset.value;
    function place(animate) {
      const btn = buttons.find((b) => b.dataset.value === current) || buttons[0];
      if (!animate) thumb.style.transition = "none";
      thumb.style.width = btn.offsetWidth + "px";
      thumb.style.transform = "translateX(" + btn.offsetLeft + "px)";
      if (!animate) { void thumb.offsetWidth; thumb.style.transition = ""; }
    }
    function set(v, animate) {
      current = v;
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.value === v)));
      place(animate !== false);
    }
    buttons.forEach((b) => {
      b.type = "button";
      b.addEventListener("click", () => { if (b.dataset.value !== current) { set(b.dataset.value); onChange && onChange(current); } });
    });
    set(current, false);
    whenLaidOut(() => place(false));
    window.addEventListener("resize", () => place(false));
    return { set, get: () => current, place };
  }

  // Navigation: items with data-page; a thumb slides to the active item and the
  // matching .page is shown with an entrance animation.
  function navigation(nav, opts) {
    const o = Object.assign({ vertical: false, onChange: null }, opts || {});
    nav.style.position = nav.style.position || "relative";
    const thumb = document.createElement("span");
    thumb.className = "nav-thumb";
    nav.prepend(thumb);
    const items = Array.from(nav.querySelectorAll("[data-page]"));
    let current = (location.hash || "").slice(1) || items[0].dataset.page;
    if (!items.some((i) => i.dataset.page === current)) current = items[0].dataset.page;
    function place(animate) {
      const item = items.find((i) => i.dataset.page === current);
      if (!item) return;
      if (!animate) thumb.style.transition = "none";
      thumb.style.width = item.offsetWidth + "px";
      thumb.style.height = item.offsetHeight + "px";
      thumb.style.transform = "translate(" + item.offsetLeft + "px," + item.offsetTop + "px)";
      if (!animate) { void thumb.offsetWidth; thumb.style.transition = ""; }
    }
    function show(page, animate) {
      current = page;
      items.forEach((i) => i.setAttribute("aria-current", i.dataset.page === page ? "page" : "false"));
      document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.dataset.page === page));
      place(animate);
      if (o.onChange) o.onChange(page);
      document.dispatchEvent(new CustomEvent("pagechange", { detail: page }));
    }
    items.forEach((i) => i.addEventListener("click", (e) => {
      e.preventDefault();
      history.replaceState(null, "", "#" + i.dataset.page);
      show(i.dataset.page, true);
    }));
    show(current, false);
    whenLaidOut(() => place(false));
    window.addEventListener("resize", () => place(false));
    return { show, place };
  }

  function whenLaidOut(fn) {
    requestAnimationFrame(() => requestAnimationFrame(fn));
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fn);
  }

  // ---------- Number formatting and tweening ----------
  function fmt(v, unit) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    return unit === "s" ? v.toFixed(3) : v.toFixed(2);
  }
  function tween(el, from, to, unit, ms) {
    const start = performance.now();
    const ease = (t) => 1 - Math.pow(1 - t, 3);
    cancelAnimationFrame(el._raf);
    function frame(now) {
      const t = Math.min(1, (now - start) / ms);
      el.firstChild.nodeValue = fmt(from + (to - from) * ease(t), unit);
      if (t < 1) el._raf = requestAnimationFrame(frame);
    }
    el._raf = requestAnimationFrame(frame);
    // Always land on the final value, even if animation frames are throttled.
    clearTimeout(el._end);
    el._end = setTimeout(() => { cancelAnimationFrame(el._raf); el.firstChild.nodeValue = fmt(to, unit); }, ms + 80);
  }

  function niceStep(max) {
    const raw = max / 4;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }

  // ---------- Tooltip ----------
  let tip;
  function tooltip() {
    if (!tip) { tip = document.createElement("div"); tip.className = "pace-tip"; tip.setAttribute("role", "status"); document.body.appendChild(tip); }
    return tip;
  }
  function showTip(x, y, lines) {
    const t = tooltip();
    t.replaceChildren();
    lines.forEach((l, i) => {
      if (i === 0) { const s = document.createElement("strong"); s.textContent = l; t.appendChild(s); return; }
      const row = document.createElement("div");
      if (l.color) { const k = document.createElement("span"); k.className = "key"; k.style.background = l.color; row.appendChild(k); }
      row.appendChild(document.createTextNode(l.text || l));
      t.appendChild(row);
    });
    const w = t.offsetWidth || 220;
    t.style.left = Math.min(window.innerWidth - w - 12, x + 14) + "px";
    t.style.top = y + 14 + "px";
    t.classList.add("show");
  }
  function hideTip() { if (tip) tip.classList.remove("show"); }

  // ---------- Race pace card ----------
  // kind "season": view (team | car), unit (% | s), statistic (mean | median),
  // baseline (measured | rebased), races (all | common).
  // kind "race": view (team | car), unit (% | s).
  function paceCard(el, kind, data) {
    const state = kind === "season"
      ? { view: "team", unit: "pct", stat: "mean", base: "rebased", races: "all" }
      : { view: "team", unit: "pct" };
    el.classList.add("pace-card");
    el.innerHTML = "";
    const controls = document.createElement("div");
    controls.className = "pace-controls";
    el.appendChild(controls);

    function addSeg(key, options, label) {
      const box = document.createElement("div");
      box.setAttribute("role", "group"); box.setAttribute("aria-label", label);
      options.forEach(([v, text]) => { const b = document.createElement("button"); b.dataset.value = v; b.textContent = text; box.appendChild(b); });
      controls.appendChild(box);
      segmented(box, state[key], (v) => { state[key] = v; render(true); });
    }
    addSeg("view", [["team", "Team race pace"], ["car", "Car only"]], "Measurement");
    addSeg("unit", [["pct", "%"], ["s", "Seconds"]], "Unit");
    if (kind === "season") {
      addSeg("stat", [["mean", "Mean"], ["median", "Median"]], "Statistic");
      addSeg("base", [["rebased", "Best car = 0"], ["measured", "As measured"]], "Baseline");
      addSeg("races", [["all", "All races"], ["common", "Common races"]], "Races");
      const help = document.createElement("span");
      help.className = "help"; help.tabIndex = 0; help.textContent = "?";
      const pop = document.createElement("span");
      pop.className = "help-pop";
      pop.textContent = "Some teams could not be measured in every race (both cars out early, or too few clean laps). " +
        "Common races uses only the races in which every team was measured, so a team that missed difficult races is not flattered.";
      help.appendChild(pop);
      controls.appendChild(help);
    }

    const canvas = document.createElement("div");
    canvas.className = "pace-canvas";
    const grid = document.createElement("div");
    grid.className = "pace-grid";
    const rowsEl = document.createElement("div");
    rowsEl.className = "pace-rows";
    canvas.append(grid, rowsEl);
    el.appendChild(canvas);

    const foot = document.createElement("div");
    foot.className = "pace-foot";
    const caption = document.createElement("span");
    const tableBtn = document.createElement("button");
    tableBtn.className = "btn"; tableBtn.type = "button"; tableBtn.textContent = "Table view";
    foot.append(caption, tableBtn);
    el.appendChild(foot);
    const tableBox = document.createElement("div");
    tableBox.hidden = true;
    el.appendChild(tableBox);
    tableBtn.addEventListener("click", () => {
      tableBox.hidden = !tableBox.hidden;
      tableBtn.textContent = tableBox.hidden ? "Table view" : "Hide table";
      renderTable();
    });

    const rows = new Map();
    let visible = false;
    let entered = false;

    function values() {
      if (kind === "season") {
        const src = data.pace[state.view][state.races];
        const key = state.stat + (state.base === "rebased" ? "_rebased" : "");
        return src.map((r) => ({ team: r.team, value: r[state.unit][key], n: r.n }));
      }
      return data.race[state.view].map((r) => {
        const value = state.unit === "pct" ? r.gap_pct : r.gap_s;
        let lo = null, hi = null;
        if (r.interval_pct) {
          const k = state.unit === "pct" ? 1 : (r.gap_pct > 0 ? r.gap_s / r.gap_pct : 0);
          lo = Math.max(0, r.interval_pct[0] * k); hi = r.interval_pct[1] * k;
        }
        return { team: r.team, value, lo, hi, drivers: r.drivers, note: r.note };
      });
    }

    function describe() {
      if (kind === "season") {
        const n = data.pace[state.view][state.races === "all" ? "all_races" : "common_races"];
        const what = state.view === "team" ? "Team race pace (car with its quicker driver)" : "Car only: driver effect removed (experimental)";
        return what + " · " + (state.stat === "mean" ? "mean" : "median") + " of per-race gaps · " + n + " races · " +
          (state.base === "rebased" ? "best car = 0" : "as measured");
      }
      return (state.view === "team" ? "Team race pace" : "Car only (" + data.race.car_condition + " laps), driver effect removed") +
        " · " + data.race.name + " " + data.season + " · bars show the 80% range";
    }

    function rowFor(r) {
      if (rows.has(r.team)) return rows.get(r.team);
      const t = team(r.team);
      const row = document.createElement("div");
      row.className = "pace-row"; row.tabIndex = 0;
      const name = document.createElement("div"); name.className = "pace-name";
      const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = t.color;
      const label = document.createElement("span"); label.textContent = t.name;
      const note = document.createElement("span"); note.className = "note";
      name.append(sw, label, note);
      const track = document.createElement("div"); track.className = "pace-track";
      const bar = document.createElement("div"); bar.className = "pace-bar" + (t.hatch ? " hatch" : "");
      bar.style.backgroundColor = t.color;
      const whisker = document.createElement("div"); whisker.className = "pace-whisker";
      track.append(bar, whisker);
      const val = document.createElement("div"); val.className = "pace-value";
      val.appendChild(document.createTextNode("0.00"));
      const unit = document.createElement("small"); val.appendChild(unit);
      row.append(name, track, val);
      const entry = { row, bar, whisker, val, unit, note, t, last: 0, data: null };
      const show = (x, y) => {
        const d = entry.data; if (!d) return;
        const u = state.unit === "pct" ? "%" : " s";
        const lines = [fmt(d.value, state.unit) + u + " behind the best car", { text: t.name, color: t.color }];
        if (d.n !== undefined) lines.push("Measured in " + d.n + " races");
        if (d.lo !== null && d.lo !== undefined) lines.push("80% range " + fmt(d.lo, state.unit) + " to " + fmt(d.hi, state.unit) + u);
        if (d.drivers && d.drivers.length) lines.push("Drivers measured: " + d.drivers.join(", "));
        if (d.note) lines.push("* Rookie in the car: measured from the established teammate only");
        showTip(x, y, lines);
      };
      row.addEventListener("pointermove", (e) => show(e.clientX, e.clientY));
      row.addEventListener("pointerleave", hideTip);
      row.addEventListener("focus", () => { const b = row.getBoundingClientRect(); show(b.left + b.width / 2, b.top); });
      row.addEventListener("blur", hideTip);
      rows.set(r.team, entry);
      return entry;
    }

    function render(update) {
      const data2 = values().sort((a, b) => a.value - b.value);
      caption.textContent = describe();
      const max = Math.max(...data2.map((d) => Math.max(d.value, d.hi || 0)), 0.01) * 1.05;
      const step = niceStep(max);
      const top = Math.ceil(max / step) * step;
      // Gridlines
      grid.replaceChildren();
      for (let v = 0; v <= top + 1e-9; v += step) {
        const tick = document.createElement("div"); tick.className = "tick";
        tick.style.left = (v / top) * 100 + "%";
        const s = document.createElement("span"); s.textContent = fmt(v, state.unit).replace(/\.?0+$/, "") + (state.unit === "pct" ? "%" : " s");
        tick.appendChild(s); grid.appendChild(tick);
      }
      // FLIP: remember where each row was.
      const before = new Map();
      rows.forEach((e, k) => before.set(k, e.row.getBoundingClientRect().top));
      const seen = new Set();
      data2.forEach((d, i) => {
        const e = rowFor(d);
        seen.add(d.team);
        e.data = d;
        rowsEl.appendChild(e.row);
        e.note.textContent = d.note ? "*" : "";
        e.unit.textContent = state.unit === "pct" ? "%" : " s";
        const width = visible ? Math.max(0.6, (d.value / top) * 100) : 0;
        e.bar.style.setProperty("--delay", (update ? i * 25 : i * 55) + "ms");
        e.bar.style.width = width + "%";
        if (d.lo !== null && d.lo !== undefined && visible) {
          e.whisker.style.left = (d.lo / top) * 100 + "%";
          e.whisker.style.width = ((d.hi - d.lo) / top) * 100 + "%";
          e.whisker.style.opacity = "0.9";
        } else { e.whisker.style.opacity = "0"; }
        if (visible) { tween(e.val, e.last, d.value, state.unit, update ? 650 : 1100); e.last = d.value; }
      });
      rows.forEach((e, k) => { if (!seen.has(k)) { e.row.remove(); rows.delete(k); } });
      // Animate rows from their old positions to the new order.
      rows.forEach((e, k) => {
        const was = before.get(k);
        if (was === undefined) return;
        const dy = was - e.row.getBoundingClientRect().top;
        if (!dy) return;
        e.row.style.transition = "none";
        e.row.style.transform = "translateY(" + dy + "px)";
        requestAnimationFrame(() => {
          e.row.style.transition = "transform 0.7s var(--spring)";
          e.row.style.transform = "";
        });
      });
      if (!tableBox.hidden) renderTable();
    }

    function renderTable() {
      if (tableBox.hidden) return;
      const d = values().sort((a, b) => a.value - b.value);
      const u = state.unit === "pct" ? "%" : "s";
      const t = document.createElement("table");
      t.className = "pace-table";
      const head = t.createTHead().insertRow();
      ["Team", "Gap (" + u + ")", kind === "season" ? "Races" : "80% range"].forEach((h) => { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); });
      const body = t.createTBody();
      d.forEach((r) => {
        const tr = body.insertRow();
        tr.insertCell().textContent = team(r.team).name + (r.note ? " *" : "");
        tr.insertCell().textContent = fmt(r.value, state.unit);
        tr.insertCell().textContent = kind === "season" ? String(r.n) : (r.lo !== null && r.lo !== undefined ? fmt(r.lo, state.unit) + " – " + fmt(r.hi, state.unit) : "–");
      });
      tableBox.replaceChildren(t);
    }

    render(false);
    // Bars grow from zero the first time the card comes into view.
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting && !entered) { entered = true; visible = true; render(false); }
      });
    }, { threshold: 0.25 });
    io.observe(el);
    // Replay the entrance when the page holding the card is shown again.
    document.addEventListener("pagechange", () => {
      if (!el.offsetParent) return;
      entered = true;
      visible = false; rows.forEach((e) => { e.bar.style.width = "0"; e.last = 0; });
      requestAnimationFrame(() => requestAnimationFrame(() => { visible = true; render(false); }));
    });
    document.addEventListener("themechange", () => { /* colours come from CSS; nothing to redraw */ });
    return { render };
  }

  // ---------- Content builders ----------
  function raceCards(el, data, opts) {
    const o = Object.assign({ limit: data.races.length, compact: false }, opts || {});
    el.replaceChildren();
    data.races.slice(0, o.limit).forEach((r, i) => {
      const card = document.createElement("article");
      card.className = "race-card";
      card.style.setProperty("--i", i);
      const head = document.createElement("header");
      const round = document.createElement("span"); round.className = "rc-round"; round.textContent = "R" + String(r.round).padStart(2, "0");
      const date = document.createElement("span"); date.className = "rc-date";
      date.textContent = new Date(r.date + "T12:00:00Z").toLocaleDateString(undefined, { day: "numeric", month: "short" });
      head.append(round, date);
      const name = document.createElement("h4"); name.textContent = r.name.replace(" Grand Prix", "");
      const place = document.createElement("p"); place.className = "rc-place"; place.textContent = r.locality + ", " + r.country;
      const pod = document.createElement("ol"); pod.className = "rc-podium";
      r.podium.forEach((p) => {
        const li = document.createElement("li");
        const t = team(p.team);
        const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = t.color;
        const code = document.createElement("b"); code.textContent = p.code;
        const nm = document.createElement("span"); nm.className = "rc-name"; nm.textContent = o.compact ? "" : p.name;
        li.append(sw, code, nm);
        pod.appendChild(li);
      });
      card.append(head, name, place, pod);
      el.appendChild(card);
    });
  }

  function standings(el, data, which, limit) {
    const list = data.standings[which].slice(0, limit || 99);
    const max = list[0].points;
    el.replaceChildren();
    list.forEach((s, i) => {
      const row = document.createElement("div");
      row.className = "st-row";
      row.style.setProperty("--i", i);
      const t = team(s.team);
      const pos = document.createElement("span"); pos.className = "st-pos"; pos.textContent = s.pos;
      const name = document.createElement("span"); name.className = "st-name";
      const sw = document.createElement("span"); sw.className = "sw"; sw.style.background = t.color;
      name.append(sw, document.createTextNode(which === "drivers" ? s.name : t.name));
      const barBox = document.createElement("span"); barBox.className = "st-bar";
      const bar = document.createElement("i"); bar.style.background = t.color; bar.style.setProperty("--w", (s.points / max) * 100 + "%");
      barBox.appendChild(bar);
      const pts = document.createElement("span"); pts.className = "st-pts"; pts.textContent = s.points;
      row.append(pos, name, barBox, pts);
      el.appendChild(row);
    });
  }

  // ---------- Placeholder sections (layout only; values are not data) ----------
  const TRACK_PATH = "M40,150 C40,90 70,60 120,58 L250,52 C300,50 330,70 336,104 C342,140 320,160 288,160 L230,160 C205,160 196,176 205,196 L214,214 C222,236 206,252 180,252 L96,252 C60,252 40,230 40,196 Z";
  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }
  function tag(text) { return el("span", "placeholder-tag", text || "Placeholder"); }

  const placeholders = {
    nextRace(box, data) {
      const r = data.races.find((x) => x.round === 22) || data.races[data.races.length - 1];
      box.replaceChildren(
        el("p", "ph-kicker", "Next race · Round " + r.round),
        el("h3", "ph-title", r.name),
        el("p", "ph-sub", r.locality + ", " + r.country + " · " + new Date(r.date + "T12:00:00Z").toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "long" })),
      );
      const cd = el("div", "ph-countdown");
      [["02", "days"], ["14", "hrs"], ["37", "min"]].forEach(([v, u]) => { const c = el("div", "cd"); c.append(el("b", "", v), el("span", "", u)); cd.appendChild(c); });
      box.append(cd, tag("Sample countdown"));
    },
    live(box) {
      box.replaceChildren(el("p", "ph-kicker", "Live session"), el("h3", "ph-title", "Practice 2"));
      const bar = el("div", "ph-live"); bar.append(el("span", "dot"), el("span", "", "Lap 18 · track green"));
      const prog = el("div", "ph-progress"); prog.appendChild(el("i"));
      box.append(bar, prog, tag("Placeholder: live timing arrives with the live service"));
    },
    strategy(box) {
      box.replaceChildren(el("p", "ph-kicker", "Possible strategies"), el("h3", "ph-title", "One stop vs two stop"));
      const tyres = [["One stop", [["M", 38], ["H", 62]]], ["Two stop", [["S", 22], ["M", 38], ["H", 40]]], ["Two stop (alt)", [["M", 30], ["H", 35], ["S", 35]]]];
      tyres.forEach(([label, stints]) => {
        const row = el("div", "ph-strat");
        row.appendChild(el("span", "lbl", label));
        const bar = el("div", "strat-bar");
        stints.forEach(([c, w], i) => { const s = el("i", "tyre tyre-" + c, c); s.style.setProperty("--w", w + "%"); s.style.setProperty("--i", i); bar.appendChild(s); });
        row.appendChild(bar);
        box.appendChild(row);
      });
      box.appendChild(tag("Illustrative layout, not a forecast"));
    },
    track(box) {
      box.replaceChildren(el("p", "ph-kicker", "Track map"), el("h3", "ph-title", "Circuit and battles"));
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 380 290"); svg.setAttribute("class", "ph-track");
      const base = document.createElementNS(ns, "path"); base.setAttribute("d", TRACK_PATH); base.setAttribute("class", "trk-base");
      const line = document.createElementNS(ns, "path"); line.setAttribute("d", TRACK_PATH); line.setAttribute("class", "trk-line");
      svg.append(base, line);
      ["#3A5BD9", "#FF8000", "#00A19C"].forEach((c, i) => {
        const dot = document.createElementNS(ns, "circle"); dot.setAttribute("r", "6"); dot.setAttribute("fill", c); dot.setAttribute("class", "trk-car");
        dot.style.offsetPath = "path('" + TRACK_PATH + "')"; dot.style.animationDelay = (-i * 0.35) + "s";
        svg.appendChild(dot);
      });
      box.append(svg, tag("Preview: the full map builds on the existing 3D track"));
    },
    calendar(box) {
      box.replaceChildren(el("p", "ph-kicker", "Calendar builder"), el("h3", "ph-title", "Build your season"));
      const globe = el("div", "ph-globe");
      for (let i = 0; i < 14; i++) { const d = el("i"); d.style.setProperty("--x", (12 + ((i * 37) % 76)) + "%"); d.style.setProperty("--y", (18 + ((i * 53) % 64)) + "%"); d.style.setProperty("--i", i); globe.appendChild(d); }
      const slots = el("ol", "ph-slots");
      ["Bahrain", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Miami"].forEach((n, i) => { const li = el("li", "", n); li.style.setProperty("--i", i); slots.appendChild(li); });
      box.append(globe, slots, tag("Placeholder"));
    },
    predictions(box) {
      box.replaceChildren(el("p", "ph-kicker", "Forecast"), el("h3", "ph-title", "Finishing position chances"));
      const heat = el("div", "ph-heat");
      for (let r = 0; r < 6; r++) for (let c = 0; c < 10; c++) {
        const cell = el("i"); const v = Math.max(0, 1 - Math.abs(c - r * 1.6) / 3.2);
        cell.style.setProperty("--v", v.toFixed(2)); cell.style.setProperty("--i", r * 10 + c); heat.appendChild(cell);
      }
      box.append(heat, tag("Illustrative layout, not a forecast"));
    },
    community(box) {
      box.replaceChildren(el("p", "ph-kicker", "Community"), el("h3", "ph-title", "From the paddock club"));
      [["calendar", "A 22-race season with back-to-back European triples", "38 likes · 12 comments"],
       ["thread", "Which corners decide the race pace at Suzuka?", "54 replies"],
       ["news", "Upgrade notes posted for the next round", "linked article"]].forEach(([k, t, m], i) => {
        const post = el("article", "ph-post"); post.style.setProperty("--i", i);
        post.append(el("span", "ph-chip", k), el("p", "", t), el("small", "", m));
        box.appendChild(post);
      });
      box.appendChild(tag("Placeholder"));
    },
    login(box) {
      box.replaceChildren(el("p", "ph-kicker", "Account"), el("h3", "ph-title", "Sign in to share calendars"));
      const f = el("form", "ph-form"); f.addEventListener("submit", (e) => e.preventDefault());
      [["Email", "email"], ["Password", "password"]].forEach(([l, t]) => { const lab = el("label", "", l); const inp = document.createElement("input"); inp.type = t; inp.placeholder = l; lab.appendChild(inp); f.appendChild(lab); });
      const b = el("button", "btn primary", "Sign in"); b.type = "submit"; f.appendChild(b);
      box.append(f, tag("Placeholder"));
    },
    summaries(box) {
      box.replaceChildren(el("p", "ph-kicker", "Season summaries"));
      const chips = el("div", "ph-chips");
      ["Drivers' championship", "Constructors' championship", "By nation", "By engine", "Poles and wins", "Fastest laps", "Retirements", "Head to head"].forEach((c, i) => {
        const b = el("button", "chip", c); b.type = "button"; b.style.setProperty("--i", i); chips.appendChild(b);
      });
      box.append(chips, tag("More summaries to come"));
    },
  };

  window.F1X = { team, TEAMS, THEMES, initTheme, applyTheme, mountThemePicker, segmented, navigation, paceCard, raceCards, standings, placeholders };
  initTheme();
})();
