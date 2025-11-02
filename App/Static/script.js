document.addEventListener("DOMContentLoaded", () => {
  const textbox = document.getElementById("textbox");
  const suggestionsDiv = document.getElementById("suggestions");

  let debounceTimer = null;
  let abortController = null;

  // Cache for faster repeated lookups
  const cache = new Map();
  const CACHE_TTL = 5000;

  function debounceFetchSuggestions(force = false) {
    clearTimeout(debounceTimer);
    if (force) return fetchSuggestions();
    debounceTimer = setTimeout(fetchSuggestions, 250);
  }

  function getCached(prefix) {
    const entry = cache.get(prefix);
    if (!entry) return null;
    if (Date.now() - entry.timestamp > CACHE_TTL) {
      cache.delete(prefix);
      return null;
    }
    return entry.data;
  }

  function setCached(prefix, data) {
    cache.set(prefix, { data, timestamp: Date.now() });
  }

  async function fetchSuggestions() {
    const text = textbox.value.trim();
    const words = text.split(/\s+/).filter(Boolean);
    const lastWord = words.length ? words[words.length - 1] : "";

    // Clear if empty input
    if (!text) {
      suggestionsDiv.innerHTML = "";
      return;
    }

    const cacheKey = text.toLowerCase();
    const cached = getCached(cacheKey);
    if (cached) {
      renderSuggestions(cached.autocorrect, cached.autocomplete, lastWord);
      return;
    }

    // Cancel any pending API calls
    if (abortController) abortController.abort();
    abortController = new AbortController();
    const { signal } = abortController;

    try {
      const [autocorrectResp, autocompleteResp] = await Promise.all([
        fetch(`/autocorrect?word=${encodeURIComponent(lastWord)}`, { signal }),
        fetch(`/autocomplete?prefix=${encodeURIComponent(text)}`, { signal }),
      ]);

      if (!autocorrectResp.ok || !autocompleteResp.ok) return;

      const autocorrectData = await autocorrectResp.json();
      const autocompleteData = await autocompleteResp.json();

      const result = {
        autocorrect: autocorrectData.suggestions || [],
        autocomplete: autocompleteData.suggestions || [],
      };

      setCached(cacheKey, result);
      renderSuggestions(result.autocorrect, result.autocomplete, lastWord);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Error fetching suggestions:", err);
        suggestionsDiv.innerHTML =
          '<p style="color:red;">Error loading suggestions</p>';
      }
    }
  }

  function renderSuggestions(autocorrectList, autocompleteList, lastWord) {
    suggestionsDiv.innerHTML = "";

    // Autocorrect suggestions (blue glow)
    autocorrectList.forEach((s) => {
      const btn = document.createElement("button");
      btn.textContent = s;
      btn.classList.add("suggestion", "autocorrect");
      btn.addEventListener("click", () => {
        const words = textbox.value.trim().split(/\s+/);
        words[words.length - 1] = s;
        textbox.value = words.join(" ") + " ";
        suggestionsDiv.innerHTML = "";
        textbox.focus();
        debounceFetchSuggestions(true); // keep autocomplete going
      });
      suggestionsDiv.appendChild(btn);
    });

    // Autocomplete suggestions (green glow)
    autocompleteList.forEach((s) => {
      const btn = document.createElement("button");
      btn.textContent = s;
      btn.classList.add("suggestion", "autocomplete");
      btn.addEventListener("click", () => {
        textbox.value = textbox.value.trim() + " " + s + " ";
        suggestionsDiv.innerHTML = "";
        textbox.focus();
        debounceFetchSuggestions(true); // predict next word immediately
      });
      suggestionsDiv.appendChild(btn);
    });
  }

  // Respond to all kinds of input (typing, deleting, pasting)
  textbox.addEventListener("input", () => debounceFetchSuggestions());

  // Additional safeguard: recheck on blur/focus
  textbox.addEventListener("blur", () => clearTimeout(debounceTimer));
  textbox.addEventListener("focus", () => debounceFetchSuggestions());
});
