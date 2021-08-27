from pygame import Surface, display, draw, image, mouse
from pygame.constants import K_RETURN, K_SPACE
from libs.neural_network import *
from PIL import Image

import pygame
pygame.init()

nn : NeuralNetwok = load_nn("model/model.npy")

running : bool = True
pressed : bool = False
previous_mouse_pos = None
my_font : pygame.font.Font = pygame.font.Font(pygame.font.get_default_font(), 20)


screen : Surface = display.set_mode((500,500))
screen.fill((0,0,0))

while running:
    previous_mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == K_SPACE:
                screen.fill((0,0,0))
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            screen.fill((0,0,0), rect=(50,0,80,50))
            pressed = False
            image.save(screen, "input.png")

            img : np.ndarray = np.array(Image.open("input.png").convert("L").resize((28, 28),Image.ANTIALIAS)).flatten() / 255

            prediction = nn.forward(img)

            text_suf_cat = my_font.render(f"cat : {int(prediction[0] * 100)}%", False, (255, 255, 255))
            text_suf_rabbit = my_font.render(f"rabbit : {int(prediction[1] * 100)}%", False, (255, 255, 255))

            screen.blit(text_suf_cat, (10,10))
            screen.blit(text_suf_rabbit, (10,30))

        

    
    if pressed:
        draw.polygon(screen, (255, 255, 255), (previous_mouse_pos, mouse.get_pos()), 10)

    display.flip()

pygame.quit()