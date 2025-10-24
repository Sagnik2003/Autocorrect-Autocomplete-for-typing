const textbox = document.getElementById("textbox");
const suggestionsDiv = document.getElementById("suggestions");

textbox.addEventListener("input", async () => {
  const text = textbox.value.trim();
  const words = text.split(" ");
  const lastWord = words[words.length - 1];

  let suggestions = [];

  if (lastWord.length > 0) {
    // Autocorrect API
    const res = await fetch(`/autocorrect?word=${lastWord}`);
    suggestions = await res.json();
  } else if (text.length > 0) {
    // Autocomplete API
    const res = await fetch(`/autocomplete?prefix=${text}`);
    suggestions = await res.json();
  }

  // Show suggestions
  suggestionsDiv.innerHTML = "";
  suggestions.forEach(s => {
    const span = document.createElement("span");
    span.className = "suggestion";
    span.textContent = s;
    span.onclick = () => {
      // Replace last word with suggestion
      words[words.length - 1] = s;
      textbox.value = words.join(" ") + " ";
      suggestionsDiv.innerHTML = "";
    };
    suggestionsDiv.appendChild(span);
  });
});
