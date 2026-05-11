"""
UNO Game Streamlit app.
Player 1 = Human (always). Player 2 = chosen AI agent.
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
from core.cards import Card, Color, CardType
from core.deck import Deck
from core.hand import Hand
from core.rules import get_valid_plays, apply_card_effect
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.bad_rule_agent import RuleBadAgent

AGENT_MAP = {
    "Random Bot":  RandomAgent,
    "Greedy Bot":   RuleAgent,
    "Anti-Greedy Bot":  RuleBadAgent,
}

COLOR_HEX = {
    Color.RED:    "#E8293B",
    Color.GREEN:  "#009A44",
    Color.BLUE:   "#0066CC",
    Color.YELLOW: "#FFD600",
    Color.WILD:   "#010114",
}
COLOR_TEXT = {
    Color.RED:    "#fff",
    Color.GREEN:  "#fff",
    Color.BLUE:   "#fff",
    Color.YELLOW: "#1A1A2E",
    Color.WILD:   "#FFD600",
}
CARD_LABELS = {
    CardType.NUMBER:         lambda c: str(c.value),
    CardType.SKIP:           lambda _: "⊘",
    CardType.REVERSE:        lambda _: "⇄",
    CardType.DRAW_TWO:       lambda _: "+2",
    CardType.WILD:           lambda _: "★",
    CardType.WILD_DRAW_FOUR: lambda _: "+4",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&family=Nunito+Sans:wght@400;600;700&display=swap');

[data-testid="stAppViewContainer"] { background:#1A1A2E !important; }
[data-testid="stHeader"]           { display:none !important; }
section[data-testid="stSidebar"]   { display:none !important; }
.block-container { padding:0.5rem 1rem !important; max-width:100% !important; }
* { box-sizing:border-box; }
body,.stMarkdown,.stText { font-family:'Nunito Sans',sans-serif !important; color:#fff !important; }

/* ── UNO card ── */
.uno-card {
  display:inline-flex; flex-direction:column; align-items:center;
  justify-content:space-between;
  width:62px; height:96px; border-radius:12px;
  border:3px solid rgba(255,255,255,.45);
  padding:4px; font-family:'Nunito',sans-serif; font-weight:900;
  text-shadow:1px 1px 3px rgba(0,0,0,.45);
  box-shadow:2px 4px 10px rgba(0,0,0,.4);
  cursor:default;
  transition:transform .15s,box-shadow .15s;
}
.uno-card .ct { font-size:12px; opacity:.9; align-self:flex-start; line-height:1; }
.uno-card .cm { font-size:24px; line-height:1; }
.uno-card .cb { font-size:12px; opacity:.9; align-self:flex-end; transform:rotate(180deg); line-height:1; }
.card-playable        { cursor:pointer !important; }
.card-playable:hover  { transform:translateY(-8px) scale(1.08) !important; box-shadow:0 8px 24px rgba(255,214,0,.35) !important; }
.card-dimmed          { opacity:.38 !important; cursor:not-allowed !important; }

/* ── Card back (small, for AI column) ── */
.card-back-sm {
  display:inline-flex; align-items:center; justify-content:center;
  width:62px; height:96px; border-radius:12px;
  background:#000000;border:3px solid rgba(255,255,255,.3);
  font-family:'Nunito',sans-serif; font-weight:900; font-size:18px; color:#FFD600;
  box-shadow:2px 4px 10px rgba(0,0,0,.35); position:relative;
}
.card-back-sm::after {
  content:''; position:absolute; inset:5px; border-radius:8px;
  border:2px solid rgba(255,255,255,.18);
}

/* ── Side panels ── */
.side-panel {
  background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
  border-radius:16px; padding:12px 10px;
}
.side-panel-label {
  font-family:'Nunito',sans-serif; font-weight:900; font-size:13px;
  color:#FFD600; text-transform:uppercase; letter-spacing:1px;
  margin-bottom:4px; text-align:center;
}
.side-panel-badge {
  display:inline-block; background:rgba(255,255,255,.12); border-radius:20px;
  padding:2px 10px; font-size:11px; font-weight:700; color:#fff; margin-bottom:8px;
}
.side-panel-badge.active { background:rgba(255,214,0,.25); color:#FFD600; border:1px solid rgba(255,214,0,.4); }

/* horizontal wrapping card row inside panels */
.h-card-wrap {
  display:flex; flex-direction:row; flex-wrap:wrap; gap:5px;
  justify-content:flex-start; align-items:flex-end; width:100%;
  padding:4px 0;
}

/* ── Center column ── */
.center-col {
  display:flex; flex-direction:column; align-items:center;
  gap:10px; padding:8px 4px;
}
.pile-title {
  font-family:'Nunito',sans-serif; font-weight:800; font-size:11px;
  text-transform:uppercase; letter-spacing:1.2px; color:rgba(255,255,255,.45);
  margin-bottom:4px; text-align:center;
}
.pile-count { font-size:11px; color:rgba(255,255,255,.4); text-align:center; }
.draw-pile-vis {
  display:inline-flex; align-items:center; justify-content:center;
  width:70px; height:108px; border-radius:13px;
  background:#000000;border:3px solid rgba(255,255,255,.3);
  font-family:'Nunito',sans-serif; font-weight:900; font-size:22px; color:#FFD600;
  box-shadow:2px 4px 10px rgba(0,0,0,.35); position:relative;
}
.draw-pile-vis::after { content:''; position:absolute; inset:6px; border-radius:8px; border:2px solid rgba(255,255,255,.18); }

/* ── Status strip ── */
.status-strip {
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.08);
  border-radius:12px; padding:8px 14px;
  display:flex; flex-direction:column; align-items:center; gap:4px;
  font-size:12px; width:100%;
}
.status-msg { font-family:'Nunito',sans-serif; font-weight:800; font-size:13px; color:#FFD600; text-align:center; }
.color-dot { display:inline-block; width:14px; height:14px; border-radius:50%; border:2px solid rgba(255,255,255,.4); vertical-align:middle; margin-right:5px; }

/* ── Wild colour picker ── */
.wild-picker { display:flex; flex-direction:column; gap:6px; width:100%; }

/* ── Streamlit button overrides ── */
div[data-testid="stButton"]>button {
  font-family:'Nunito',sans-serif !important; font-weight:800 !important;
  border-radius:50px !important; transition:transform .1s !important; font-size:13px !important;
}
div[data-testid="stButton"]>button:hover  { transform:translateY(-1px) !important; }
div[data-testid="stButton"]>button:active { transform:scale(.96) !important; }

/* ── Select / input ── */
div[data-baseweb="select"]>div,
div[data-baseweb="input"]>div>input {
  background:rgba(255,255,255,.08) !important; border-color:rgba(255,255,255,.2) !important;
  color:#fff !important; border-radius:10px !important;
}
label[data-testid="stWidgetLabel"] { color:rgba(255,255,255,.7) !important; }
hr { border-color:rgba(255,255,255,.08) !important; margin:6px 0 !important; }

/* ── Game over ── */
.go-banner { text-align:center; padding:2.5rem; background:rgba(255,214,0,.07); border:2px solid rgba(255,214,0,.22); border-radius:20px; margin:1rem 0; }
.go-title  { font-family:'Nunito',sans-serif; font-weight:900; font-size:44px; }
.go-sub    { font-size:16px; color:rgba(255,255,255,.6); margin-top:4px; }

/* ── Log ── */
.log-entry { font-size:11px; color:rgba(255,255,255,.5); padding:2px 0; border-bottom:1px solid rgba(255,255,255,.05); }
.log-entry:last-child { color:rgba(255,255,255,.88); font-weight:700; }

/* ── Thinking indicator ── */
.thinking-dot { animation: blink 1s infinite; display:inline-block; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
</style>
"""

def _lbl(card: Card) -> str:
    return CARD_LABELS[card.card_type](card)


def _card_html(card: Card, playable: bool = False, override_color: Color | None = None) -> str:
    c   = override_color or card.color
    bg  = COLOR_HEX[c]
    txt = COLOR_TEXT[c]
    lbl = _lbl(card)
    bdr = "rgba(0,0,0,.4)" if c == Color.YELLOW else "rgba(255,255,255,.45)"
    # cls = "uno-card" + (" card-playable" if playable else " card-dimmed")
    cls = "uno-card" + (" card-playable" if playable else "")
    return (
        f'<div class="{cls}" style="background:{bg};color:{txt};border-color:{bdr};">'
        f'<span class="ct">{lbl}</span><span class="cm">{lbl}</span><span class="cb">{lbl}</span>'
        f'</div>'
    )


def _get_valid(hand: Hand, deck: Deck, color: Color):
    return get_valid_plays(hand.cards, deck.top_card(), color)


def _init_game(ai_name: str, ai_cls) -> None:
    deck = Deck()
    human_hand = Hand([deck.draw() for _ in range(7)])
    ai_hand    = Hand([deck.draw() for _ in range(7)])
    deck.setup_first_card()
    st.session_state.update(
        page="game",
        human_name="You",
        ai_name=ai_name,
        ai_agent=ai_cls(ai_name),
        deck=deck,
        human_hand=human_hand,
        ai_hand=ai_hand,
        current_color=deck.top_card().color,
        human_turn=True,
        turn_count=1,
        game_log=[f"Game started: You vs {ai_name}"],
        winner=None,
        pending_wild=None,
        drawn_this_turn=False,
    )


def _execute_human_play(card: Card, chosen_color: Color | None) -> None:
    s = st.session_state
    s.human_hand.remove(card)
    s.deck.discard(card)
    effect = apply_card_effect(card, chosen_color)
    s.current_color = effect["new_color"]
    note = f" → chose {chosen_color.value}" if chosen_color else ""
    s.game_log.append(f"✋ You played {card.color.value} {_lbl(card)}{note}")
    s.pending_wild    = None
    s.drawn_this_turn = False

    if s.human_hand.is_empty():
        s.winner = "human"
        return

    if effect["opponent_draws"] > 0:
        for _ in range(effect["opponent_draws"]):
            s.ai_hand.add(s.deck.draw())
        s.game_log.append(f"💥 {s.ai_name} draws {effect['opponent_draws']} cards!")

    if effect["skip_opponent"]:
        s.game_log.append(f"⏭ {s.ai_name}'s turn skipped!")
        s.turn_count += 1
        s.human_turn = True
        return

    s.human_turn = False
    s.turn_count += 1
    _ai_turn()


def _ai_turn() -> None:
    s     = st.session_state
    valid = _get_valid(s.ai_hand, s.deck, s.current_color)

    if not valid:
        s.ai_hand.add(s.deck.draw())
        s.game_log.append(f"🤖 {s.ai_name} drew a card")
        s.human_turn     = True
        s.turn_count    += 1
        s.drawn_this_turn = False
        return

    from core.game import GameState
    state = GameState(
        current_player=1,
        top_card=s.deck.top_card(),
        current_color=s.current_color,
        hand_sizes=[s.human_hand.size(), s.ai_hand.size()],
        draw_pile_size=s.deck.draw_pile_size,
        valid_plays=valid,
    )
    card, chosen_color = s.ai_agent.choose_action(state)
    s.ai_hand.remove(card)
    s.deck.discard(card)
    effect = apply_card_effect(card, chosen_color)
    s.current_color = effect["new_color"]
    note = f" → chose {chosen_color.value}" if chosen_color else ""
    s.game_log.append(f"🤖 {s.ai_name} played {card.color.value} {_lbl(card)}{note}")

    if s.ai_hand.is_empty():
        s.winner = "ai"
        return

    if effect["opponent_draws"] > 0:
        for _ in range(effect["opponent_draws"]):
            s.human_hand.add(s.deck.draw())
        s.game_log.append(f"💥 You draw {effect['opponent_draws']} cards!")

    if effect["skip_opponent"]:
        s.game_log.append("⏭ Your turn was skipped!")
        s.turn_count += 1
        _ai_turn()
        return

    s.human_turn     = True
    s.turn_count    += 1
    s.drawn_this_turn = False

def _page_setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1.5rem;">'
        '<span style="font-family:Nunito,sans-serif;font-weight:900;font-size:96px;'
        'color:#FFD600;text-shadow:6px 6px 0 #E8293B,12px 12px 0 rgba(0,0,0,.2);'
        'letter-spacing:-3px;display:inline-block;transform:rotate(-3deg)">UNO</span></div>',
        unsafe_allow_html=True,
    )
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="uno-panel" style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:14px 16px;margin-bottom:8px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Nunito,sans-serif;font-weight:800;font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:rgba(255,255,255,.45);margin-bottom:6px;">Choose your opponent</div>', unsafe_allow_html=True)
        ai_choice = st.selectbox("", list(AGENT_MAP.keys()), label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("🃏  DEAL CARDS", use_container_width=True, type="primary"):
            _init_game(ai_choice, AGENT_MAP[ai_choice])
            st.rerun()


def _page_game() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    s = st.session_state

    deck:       Deck = s.deck
    human_hand: Hand = s.human_hand
    ai_hand:    Hand = s.ai_hand
    top_card         = deck.top_card()
    cur_color        = s.current_color
    valid_cards      = _get_valid(human_hand, deck, cur_color) if s.human_turn else []
    if s.winner:
        human_won = s.winner == "human"
        st.markdown(
            f'<div class="go-banner">'
            f'<div style="font-size:68px">{"🎉" if human_won else "😔"}</div>'
            f'<div class="go-title" style="color:{"#FFD600" if human_won else "#E8293B"}">'
            f'{"You Win!" if human_won else "You Lose!"}</div>'
            f'<div class="go-sub">{"Won in "+str(s.turn_count)+" turns!" if human_won else s.ai_name+" wins this time!"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄  Play Again", use_container_width=True, type="primary"):
                _init_game(s.ai_name, type(s.ai_agent))
                st.rerun()
        with c2:
            if st.button("🏠  Main Menu", use_container_width=True):
                st.session_state.page = "setup"
                st.rerun()
        with st.expander("Game Log"):
            for e in reversed(s.game_log[-40:]):
                st.markdown(f'<div class="log-entry">{e}</div>', unsafe_allow_html=True)
        return

    # AI hand on the left, human hand on the right, everything else in the center
    left_col, center_col, right_col = st.columns([3, 2, 3])

    # Left panel: AI hand (card backs only, with count) + name + thinking indicator
    with left_col:
        ai_active = not s.human_turn
        badge_cls = "side-panel-badge active" if ai_active else "side-panel-badge"
        thinking  = ' ●●●' if ai_active else ""
        show_n = min(ai_hand.size(), 20)
        backs  = "".join('<div class="card-back-sm">UNO</div>' for _ in range(show_n))
        overflow = f'<div style="font-size:11px;color:rgba(255,255,255,.4);padding:4px">+{ai_hand.size()-20} more</div>' if ai_hand.size() > 20 else ""
        st.markdown(
            f'<div class="side-panel">'
            f'<div class="side-panel-label">🤖 {s.ai_name}{thinking}</div>'
            f'<div style="text-align:center;margin-bottom:6px"><span class="{badge_cls}">{ai_hand.size()} cards</span></div>'
            f'<div class="h-card-wrap">{backs}{overflow}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Pile, Status and Controls in the center
    with center_col:
        # Status strip
        if s.pending_wild:
            msg = "Wild! Pick a colour ↓"
        elif s.human_turn:
            msg = "Click a card to play!" if valid_cards else "No plays draw a card!"
        else:
            msg = f"{s.ai_name} is thinking…"

        dot = COLOR_HEX[cur_color]
        st.markdown(
            f'<div class="status-strip">'
            f'<div><span class="color-dot" style="background:{dot}"></span>'
            f'<span style="font-size:11px;color:rgba(255,255,255,.55)">Color: '
            f'<b style="color:{dot}">{cur_color.value.upper()}</b> &nbsp;·&nbsp; Turn {s.turn_count}</span></div>'
            f'<div class="status-msg">{msg}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Wild colour picker
        if s.pending_wild:
            st.markdown(
                '<div style="text-align:center;padding:4px 0 2px">'
                '<span style="font-family:Nunito,sans-serif;font-weight:800;font-size:11px;'
                'text-transform:uppercase;letter-spacing:1.2px;color:rgba(255,255,255,.45);">Choose a colour</span></div>',
                unsafe_allow_html=True,
            )
            pw1, pw2, pw3, pw4 = st.columns(4)
            wild_opts = [
                (pw1, Color.RED,    "🔴"),
                (pw2, Color.BLUE,   "🔵"),
                (pw3, Color.GREEN,  "🟢"),
                (pw4, Color.YELLOW, "🟡"),
            ]
            for wcol, cval, cname in wild_opts:
                with wcol:
                    if st.button(cname, key=f"wc_{cval.value}", use_container_width=True):
                        _execute_human_play(s.pending_wild, cval)
                        st.rerun()
            st.markdown("---")

        # Draw pile + discard side by side in center
        discard_override = cur_color if top_card.is_wild() else None
        top_html = _card_html(top_card, playable=False, override_color=discard_override)
        st.markdown(
    f'<div style="display:flex;justify-content:center;align-items:center;gap:52px;padding:18px 0 8px;">'

    # Draw pile (smaller + dimmer)
    f'<div style="display:flex;flex-direction:column;align-items:center;gap:6px;opacity:.55;transform:scale(.9);">'
    f'  <span class="pile-title">Draw Pile</span>'
    f'  <div class="draw-pile-vis" style="transform:scale(.92);">UNO</div>'
    f'  <span class="pile-count">{deck.draw_pile_size} cards</span>'
    f'</div>'

    # Main active card (larger + glowing)
    f'<div style="display:flex;flex-direction:column;align-items:center;gap:10px;">'
    f'  <span class="pile-title" style="color:#FFD600;font-size:13px;">Card In Play</span>'
    f'  <div style="transform:scale(1.32);filter:drop-shadow(0 0 24px rgba(255,214,0,.55));">'
    f'    {top_html}'
    f'  </div>'
    f'</div>'

    f'</div>',
    unsafe_allow_html=True,
)

        # Draw Card button
        draw_disabled = not s.human_turn or bool(s.drawn_this_turn) or bool(s.pending_wild)
        _, draw_btn_col, _ = st.columns([1, 2, 1])
        with draw_btn_col:
            if st.button(
                "🃏  Draw Card",
                disabled=draw_disabled,
                use_container_width=True,
                key="draw_btn",
            ):
                drawn = deck.draw()
                human_hand.add(drawn)
                s.drawn_this_turn = True
                s.game_log.append("✋ You drew a card")
                if not _get_valid(human_hand, deck, cur_color):
                    s.human_turn      = False
                    s.turn_count     += 1
                    s.drawn_this_turn = False
                    _ai_turn()
                st.rerun()

        # Recent log
        log_html = "".join(f'<div class="log-entry">{e}</div>' for e in reversed(s.game_log[-5:]))
        st.markdown(
            f'<div style="background:rgba(0,0,0,.25);border-radius:10px;padding:8px 12px;margin:8px 0 4px;">{log_html}</div>',
            unsafe_allow_html=True,
        )

        # Quit + full log
        _, q_col = st.columns([3, 1])
        with q_col:
            if st.button("✕ Quit", use_container_width=True, key="quit_btn"):
                st.session_state.page = "setup"
                st.rerun()
        with st.expander("📜 Full Log", expanded=False):
            for e in reversed(s.game_log[-60:]):
                st.markdown(f'<div class="log-entry">{e}</div>', unsafe_allow_html=True)
    # Right panel: Human hand (with playable cards highlighted) + name + active indicator + play buttons
    with right_col:
        h_active  = s.human_turn
        badge_cls = "side-panel-badge active" if h_active else "side-panel-badge"
        hint = " 🟡" if (h_active and valid_cards and not s.pending_wild) else ""

        cards = human_hand.cards
        playable_flags = [h_active and c in valid_cards and not s.pending_wild for c in cards]

        # Render all cards as one HTML block (horizontal wrap)
        cards_html = "".join(
            _card_html(c, playable=playable_flags[i]) for i, c in enumerate(cards)
        )
        st.markdown(
            f'<div class="side-panel">'
            f'<div class="side-panel-label">✋ You{hint}</div>'
            f'<div style="text-align:center;margin-bottom:6px"><span class="{badge_cls}">{human_hand.size()} cards</span></div>'
            f'<div class="h-card-wrap">{cards_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Play buttons rendered below the panel using st.columns
        # Only show buttons for playable cards
        playable_indices = [i for i, p in enumerate(playable_flags) if p]
        if playable_indices:
            st.markdown(
                '<div style="font-family:Nunito,sans-serif;font-weight:800;font-size:10px;'
                'text-transform:uppercase;letter-spacing:1px;color:rgba(255,255,255,.4);'
                'margin:6px 0 3px;text-align:center;">Tap to play</div>',
                unsafe_allow_html=True,
            )
            # Show up to 8 play buttons per row, with color dot
            COLOR_EMOJI = {
                Color.RED:    "🔴",
                Color.GREEN:  "🟢",
                Color.BLUE:   "🔵",
                Color.YELLOW: "🟡",
                Color.WILD:   "⭐",
            }
            btn_rows = [playable_indices[i:i+8] for i in range(0, len(playable_indices), 8)]
            for btn_row in btn_rows:
                row_cols = st.columns(len(btn_row))
                for col, idx in zip(row_cols, btn_row):
                    card = cards[idx]
                    lbl  = _lbl(card)
                    dot  = COLOR_EMOJI[card.color]
                    with col:
                        if st.button(f"{dot} {lbl}", key=f"card_{idx}", use_container_width=True):
                            if card.is_wild():
                                s.pending_wild = card
                                st.rerun()
                            else:
                                _execute_human_play(card, None)
                                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="UNO",
        page_icon="🃏",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if "page" not in st.session_state:
        st.session_state.page = "setup"
    if st.session_state.page == "setup":
        _page_setup()
    else:
        _page_game()


if __name__ == "__main__":
    main()