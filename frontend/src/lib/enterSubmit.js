// Enter-to-submit helpers for chat/input boxes.
//
// Many pages wire Enter on a <textarea>/<input> to send a message or add a
// row. Done naively (`e.key === 'Enter'`), this fires while an IME is still
// composing a character — in Chinese/Japanese/Korean input, Enter is the
// confirm-candidate key, not a submit. The user then watches half-formed
// pinyin get posted as a message.
//
// `isImeEnter(e)` returns true when the Enter keystroke is owned by the IME
// and therefore MUST NOT trigger submission. Callers pair it with their own
// shiftKey / preventDefault logic.
//
//   onKeyDown={e => { if (e.key === 'Enter' && !isImeEnter(e)) send() }}
//
// `e.nativeEvent?.isComposing` is the React-wrapped path; `e.isComposing`
// covers plain DOM events; `keyCode === 229` is the Safari fallback — WebKit
// bug 165004 delivers compositionend before the confirming Enter keydown, so
// `isComposing` is already false by the time this handler runs, but the
// browser still reports VK_PROCESSKEY (229) on the confirming keystroke.
export function isImeEnter(e) {
  return Boolean(
    e.nativeEvent?.isComposing || e.isComposing || e.keyCode === 229
  )
}
