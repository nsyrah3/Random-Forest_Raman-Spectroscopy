import pandas as pd
import matplotlib.pyplot as plt

# Load the Raman spectroscopy data
data = pd.read_csv('raman_data.csv')

# Extract the relevant columns
wavelength = data['Wavelength']  # Replace with the actual column name for wavelengths
intensity = data['Intensity']      # Replace with the actual column name for intensity

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(wavelength, intensity, color='blue')
plt.title('Raman Spectroscopy Data')
plt.xlabel('Wavelength (nm)')  # Adjust the label based on your data
plt.ylabel('Intensity (a.u.)')  # Adjust the label based on your data
plt.grid()
plt.xlim([wavelength.min(), wavelength.max()])  # Set x-axis limits
plt.ylim([0, intensity.max() * 1.1])  # Set y-axis limits
plt.show()