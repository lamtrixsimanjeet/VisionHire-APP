import os  # accessing the os functions
import check_camera
import Capture_Image
import Train_Image
import Recognize
from tkinter import * 
import tkinter as tk
import matplotlib.pyplot as plt
import threading
from PIL import ImageTk, Image



# creating the title bar function


# ---------------------------------------------------------
# calling the camera test function from check camera.py file

def cc_call():
	tkStatus.set("Accessing Camera...")
	status_label.update()
	check_camera.camer()
	tkStatus.set("")
	status_label.update()

def checkCamera():
	t1=threading.Thread(target=cc_call,daemon=True)
	t1.start()
	
def autom_call():
	tkStatus.set("Sending Mail...")
	status_label.update()
	os.system("py automail.py")
	tkStatus.set("Mail Sent")
	status_label.update()

def autom():
	t5=threading.Thread(target=autom_call,daemon=True)
	t5.start()


# --------------------------------------------------------------
# calling the take image function form capture image.py file

def cfaces_call():
    tkStatus.set("Capturing Faces...")
    status_label.update()
    try:
        Capture_Image.takeImages(str(tkID.get()),str(tkName.get()),str(tkEmail.get()))
        tkStatus.set("Faces Captured")
        status_label.update()
        tkID.set("")
        id_label.update()
        tkName.set("")
        name_label.update()
        tkEmail.set("")
        email_label.update()
    except Exception as e:
        if (tkID.get().isnumeric()==False):
            tkStatus.set("Enter valid user id")
            status_label.update()
        else:
            tkStatus.set("Enter valid user email")
            status_label.update()


def CaptureFaces():
	t2=threading.Thread(target=cfaces_call,daemon=True)
	t2.start()

# -----------------------------------------------------------------
# calling the train images from train_images.py file

def timages_call():
	tkStatus.set("Training Images...")
	status_label.update()
	Train_Image.TrainImages()
	tkStatus.set("Images Trained.")
	status_label.update()

def Trainimages():
	t3=threading.Thread(target=timages_call,daemon=True)
	t3.start()
	


# --------------------------------------------------------------------
# calling the recognize_attendance from recognize.py file
def rfaces_call():
	tkStatus.set("Strating...")
	status_label.update()
	Recognize.recognize_attendence()
	tkStatus.set("Completed...")
	status_label.update()

def RecognizeFaces():
	t4=threading.Thread(target=rfaces_call,daemon=True)
	t4.start()


# ---------------main driver ------------------
# create a tkinter window
root = Tk()  
# Open window having dimension 500x575
 
root.geometry("500x575")
#root.configure(bg='blue')
#Designing APP outlook
img =Image.open('tull.jpeg')
bg = ImageTk.PhotoImage(img)
label = Label(root, image=bg)
label.place(x = 0,y = 0)
#Vision Hire Background
photo = PhotoImage(file = "drcico.png")
root.iconphoto(False, photo)


root.title("VISION HIRE")
tkID = tk.StringVar()
tkName = tk.StringVar()
tkEmail = tk.StringVar()  
tkStatus = tk.StringVar()      
 

# Create a Button

btn1 = tk.Button(
    root,
    text='CHECK CAMERA',
    command=checkCamera,
    width=42,
    bg='#beef00',
    fg='#1400c6',
    bd=2,
    relief=tk.FLAT,
    activebackground = "Green",
    activeforeground = "White",
    )
btn1.grid(
    padx=15,
    pady=8,
    ipadx=84,
    ipady=16,
    row=0,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

id_label = tk.Label(root,
        text='Enter ID:', bg='#3AAFA9',
        anchor=tk.W)
id_label.grid(
    padx=12,
    pady=(8, 0),
    ipadx=0,
    ipady=1,
    row=1,
    column=0,
    columnspan=1,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

id_entry = tk.Entry(root, textvariable=tkID,
                           bg='#f5e6cc', exportselection=0,
                           relief=tk.FLAT)
id_entry.grid(
    padx=15,
    pady=6,
    ipadx=8,
    ipady=8,
    row=1,
    column=1,
    columnspan=3,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

name_label = tk.Label(root,
        text='Enter Name:', bg='#3AAFA9',
        anchor=tk.W)
name_label.grid(
    padx=12,
    pady=(8, 0),
    ipadx=0,
    ipady=1,
    row=2,
    column=0,
    columnspan=1,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

name_entry = tk.Entry(root, textvariable=tkName,
                           bg='#f5e6cc', exportselection=0,
                           relief=tk.FLAT)
name_entry.grid(
    padx=15,
    pady=6,
    ipadx=8,
    ipady=8,
    row=2,
    column=1,
    columnspan=3,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

email_label = tk.Label(root,
        text='Enter Email:', bg='#3AAFA9',
        anchor=tk.W)
email_label.grid(
    padx=12,
    pady=(8, 0),
    ipadx=0,
    ipady=1,
    row=3,
    column=0,
    columnspan=1,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

email_entry = tk.Entry(root, textvariable=tkEmail,
                           bg='#f5e6cc', exportselection=0,
                           relief=tk.FLAT)
email_entry.grid(
    padx=15,
    pady=6,
    ipadx=8,
    ipady=8,
    row=3,
    column=1,
    columnspan=3,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )


btn2 = tk.Button(
    root,
    text='CAPTURE FACES',
    command=CaptureFaces,
    width=42,
    bg='#beef00',
    fg='#1400c6',
    bd=2,
    relief=tk.FLAT,
    activebackground = "Green",
    activeforeground = "White",
    )
btn2.grid(
    padx=15,
    pady=8,
    ipadx=84,
    ipady=16,
    row=4,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )


btn3 = tk.Button(
    root,
    text='TRAIN IMAGES',
    command=Trainimages,
    width=42,
    bg='#beef00',
    fg='#1400c6',
    bd=2,
    relief=tk.FLAT,
    activebackground = "Green",
    activeforeground = "White",
    )
btn3.grid(
    padx=15,
    pady=8,
    ipadx=84,
    ipady=16,
    row=5,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )


btn4 = tk.Button(
    root,
    text='START INTERVIEW',
    command=RecognizeFaces,
    width=42,
    bg='#beef00',
    fg='#1400c6',
    bd=2,
    relief=tk.FLAT,
    activebackground = "Green",
    activeforeground = "White",
    )
btn4.grid(
    padx=15,
    pady=8,
    ipadx=84,
    ipady=16,
    row=6,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )


btn6 = tk.Button(
    root,
    text='EXIT',
    command=root.destroy,
    width=42,
    bg='#beef00',
    fg='#1400c6',
    bd=2,
    relief=tk.FLAT,
    activebackground = "Green",
    activeforeground = "White",
    )
btn6.grid(
    padx=15,
    pady=8,
    ipadx=84,
    ipady=16,
    row=8,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )

status_label = tk.Label(
    root,
    textvariable=tkStatus,
    bg='#FFCB9A',
    anchor=tk.W,
    justify=tk.LEFT,
    relief=tk.FLAT,
    wraplength=350,
    )
status_label.grid(
    padx=12,
    pady=(0, 12),
    ipadx=0,
    ipady=1,
    row=9,
    column=0,
    columnspan=4,
    sticky=tk.W + tk.E + tk.N + tk.S,
    )
 
# Set the position of button on the top of window.      
 
root.mainloop()
#mainMenu()
