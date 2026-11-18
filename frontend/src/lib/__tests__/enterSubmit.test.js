// Run with:  node --test frontend/src/lib/__tests__/enterSubmit.test.js
// (no bundler needed — enterSubmit.js is a plain ES module with no imports.)

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { isImeEnter } from '../enterSubmit.js'

test('plain Enter is not an IME confirm', () => {
  assert.equal(isImeEnter({ key: 'Enter' }), false)
})

test('React synthetic event with nativeEvent.isComposing counts as IME', () => {
  assert.equal(
    isImeEnter({ key: 'Enter', nativeEvent: { isComposing: true } }),
    true
  )
})

test('plain DOM event with isComposing counts as IME', () => {
  assert.equal(isImeEnter({ key: 'Enter', isComposing: true }), true)
})

test('Safari fallback: keyCode 229 counts as IME even when isComposing is false', () => {
  assert.equal(
    isImeEnter({ key: 'Enter', isComposing: false, keyCode: 229 }),
    true
  )
})

test('normal keyCode 13 is not an IME confirm', () => {
  assert.equal(isImeEnter({ key: 'Enter', keyCode: 13 }), false)
})
