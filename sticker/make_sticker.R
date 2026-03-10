# File to make and adjust the hexagonal sticker

library(hexSticker)
library(magick)
library(sysfonts)
library(tidyverse)

setwd("/Users/tijnjacobs/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/Documents/GitHub/FusionForests/sticker")

fusion_img <- image_read("intertwined.png")

fonts_df <- font_files()

font_add("Avenir Next", "Avenir Next.ttc")
font_add("Avenir Next Condensed", "Avenir Next Condensed.ttc")
font_add("DIN Condensed", "DIN Condensed Bold.ttf")
font_add("DIN Alternate", "DIN Alternate Bold.ttf")
font_add("Helvetica Neue", "HelveticaNeue.ttc")


sticker(
  
  # package title
  package = "FusionForests",
  p_size = 30,
  p_y = 0.55,
  
  # border thickness
  h_size = 1.4,
  
  # tree logo (slightly larger)
  subplot = fusion_img,
  s_x = 1,
  s_y = 1.12,
  s_width = 1.65,
  s_height = 1.65,
  
  # colours inspired by the logo palette
  h_fill = "#7FC8B2",
  h_color = "#2F5F56",
  p_color = "#2F5F56",
  
  # spotlight
  spotlight = TRUE,
  l_x = 1.35,
  l_y = 1.55,
  l_width = 2,
  l_alpha = 0.22,
  
  filename = "FusionForests_hex.png",
  dpi = 600,
  p_family = "DIN Alternate"
)