<script setup>
import ImageUploader from './components/ImageUploader.vue'
</script>

<template>
  <div>
    <ImageUploader />
  </div>
</template>

<style scoped>
/* No changes made to the style section */
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
