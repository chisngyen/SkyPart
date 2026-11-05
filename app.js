/* SkyPart project page: BibTeX copy and click-to-enlarge figures. No build step. */
(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);

  function initCopy() {
    const button = $('#copyButton');
    const status = $('#copyStatus');
    const code = $('#bibCode');
    let resetTimer;
    button.addEventListener('click', async () => {
      clearTimeout(resetTimer);
      status.textContent = '';
      $('span', button).textContent = 'Copy';
      button.disabled = true;
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(code.textContent);
        // A success label is never shown before the clipboard promise resolves.
        $('span', button).textContent = 'Copied';
        status.textContent = 'Copied';
        resetTimer = setTimeout(() => { $('span', button).textContent = 'Copy'; status.textContent = ''; }, 2400);
      } catch (_) {
        $('#bibBlock').focus({ preventScroll: true });
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Press Ctrl+C';
      } finally { button.disabled = false; }
    });
  }
  initCopy();
  initLightbox();
})();

// Click a figure: it fills the screen on black; click anywhere or press Escape to close.
function initLightbox() {
  const box = document.getElementById('lightbox');
  if (!box) return;
  const img = box.querySelector('img');
  let opener = null;
  const close = () => { box.classList.remove('is-open'); document.body.style.overflow = ''; setTimeout(() => { box.hidden = true; }, 200); if (opener) opener.focus({ preventScroll: true }); };
  document.querySelectorAll('.zoomable').forEach(el => {
    const show = () => {
      const src = el.querySelector('img');
      opener = el; img.src = src.currentSrc || src.src; img.alt = src.alt;
      box.hidden = false; document.body.style.overflow = 'hidden';
      requestAnimationFrame(() => box.classList.add('is-open'));
      box.querySelector('.lb-close').focus({ preventScroll: true });
    };
    el.addEventListener('click', show);
    el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(); } });
  });
  box.addEventListener('click', close);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !box.hidden) close(); });
}
