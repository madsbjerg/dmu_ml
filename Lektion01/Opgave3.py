from tkinter import *

window = Tk()

def quit_command():
    quit()

def change():
    print("Change text clicked.")

#frame = Frame(window)
#frame.pack()

lbl = Label(window, text="Hej velkommen til min GUI hihi\n\n\n", font=("Arial Bold", 15))
lbl.grid(column=0, row=0)


btn_quit = Button(window,
                  text="QUIT",
                  font=("Arial Bold", 15),
                  fg="#880808",
                  command=quit_command)
btn_quit.grid(column=0, row=1)


btn_change = Button(window,
                    text="Change Text",
                    font=("Arial Bold", 15),
                    command=change)
btn_change.grid(column=0, row=2)

window.mainloop()

