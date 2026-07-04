// Work around a <model-viewer> v4 quirk: an element parsed from HTML with a
// pre-set `src`, upgraded while below the fold, can fail to start loading even
// with loading="eager". Assigning `src` after the element is defined (the path
// a dynamically-created viewer takes) loads reliably — so ship `data-src` and
// promote it here once the custom element is registered.
customElements.whenDefined("model-viewer").then(() => {
  for (const mv of document.querySelectorAll("model-viewer[data-src]")) {
    mv.setAttribute("src", mv.dataset.src);
    mv.removeAttribute("data-src");
  }
});
