import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
	try:
		df = pd.read_csv("data/data.csv")
		df.plot.scatter("km", "price")
		plt.show()
	except:
		"An error occured whilst trying to display the data.\nMake sure the csv is at data/data.csv and tkinter is installed on your machine"