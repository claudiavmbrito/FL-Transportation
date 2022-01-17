from pathlib import Path
import sqlite3
from sqlite3 import Connection

from standards import EuroStandards

class EmissionModels:

  def __init__(self, vmean, distance):
    self.vmean=vmean
    self.distance=distance

  def get_standard(self, year, segment):
      return EuroStandards.find_standard(year, segment)

  def emissions_calculation_fullinfo(self, vmean, pollutant, fuel=None, category=None, distance=None, year=None):
    URI_SQLITE_DB = "emissions.db"

    #Get the database
    conn = sqlite3.connect(URI_SQLITE_DB)
    #Create table 
    conn.execute('''
              CREATE TABLE IF NOT EXISTS RoadEmission(Category text, 
                        Fuel text, 
                        Segment text,
                        Euro_Standard text,
                        Technology text,
                        Pollutant text, 
                        Road_Slope double,
                        Load double,
                        Min_Speed double,
                        Max_Speed double,
                        Alpha double,
                        Beta double,
                        gamma double,
                        Delta double,
                        Epsilon double,
                        Zita double, 
                        Hta double,  
                        Redu_factor double);''')

    conn.commit()
    
    #limiting the velocity of the transportation
    vmax = 130
    vmin = 10
    euro = self.get_standard(year, category)
    #the segment represents the size of the vehicle
    segment = 'Medium'
    tech = "GDI"

    if vmean > vmax:
      vmean = vmax
    else:
      vmean=vmean

    if vmin < vmin:
      vmean = vmin
    else: 
      vmean=vmean

    alpha = 0
    beta = 0
    gamma = 0
    delta = 0
    epsilon = 0
    zita = 0
    hta = 0
    redu_factor = 0
    #selecting the values from the table
    cur = conn.execute("SELECT DISTINCT alpha, beta, gamma, delta, epsilon, zita, hta, redu_factor from RoadEmission where Category ==? and Pollutant==? and Fuel==? and Euro_Standard==? and Segment==? and Technology=?", (category, pollutant, fuel, euro, segment, tech))

    for row in cur:
      print(row)
      alpha = row[0]
      beta = row[1]
      gamma = row[2]
      delta = row[3]
      epsilon = row[4]
      zita = row[5]
      hta = row [6]
      redu_factor = row[7]
    
    if (epsilon*vmean**2+zita*vmean+hta)==0:
      emission = 0

    else:
      emission = ((alpha*vmean**2+beta*vmean+gamma+(delta/vmean))/(epsilon*vmean**2+zita*vmean+hta)*(1-redu_factor))
      emission = emission * distance

    # for the co2 emissions, other calculations are needed
    if pollutant == "EC":
      if fuel == 'Petrol':
        emission = (emission/43.774)*3.169
      elif fuel == 'Diesel':
        emission = (emission/42.695)*3.169
      elif fuel == 'LPG':
        emission = (emission/46.564)*3.024

    return emission

  def full_report(self, vmean, fuel=None, category=None, distance=None, year=None):
    pol_co='CO'
    pol_pm='PM Exhaust'
    pol_no='NOx'
    pol_ec='EC'
    pol_ch='CH4'
    pol_voc='VOC'

    if category == 'motorcycle':
        return print("A motorcycle is being used, it is not possible to calculate the emissions at this moment.")
        
    else: 
        co = self.emissions_calculation_fullinfo(vmean, pol_co, fuel, category, distance, year)
        nox = self.emissions_calculation_fullinfo(vmean, pol_no, fuel, category, distance, year)
        pm = self.emissions_calculation_fullinfo(vmean, pol_pm, fuel, category, distance, year)
        co2 = self.emissions_calculation_fullinfo(vmean, pol_ec, fuel, category, distance, year)
        ch4 = self.emissions_calculation_fullinfo(vmean, pol_ch, fuel, category, distance, year)
        voc = self.emissions_calculation_fullinfo(vmean, pol_voc, fuel, category, distance, year)
        
        return co, nox, pm, co2, ch4, voc

    

  def emissions_calculation(self, vmean, fuel_type=None):
  #limiting the velocity of the transportation
  #only works for CO2
    vmax = 130
    vmin = 10

    if vmean > vmax:
      vmean = 130
    else:
      vmean=vmean

    if vmin < vmin:
      vmean = vmin
    else: 
      vmean=vmean

    if fuel_type == 'diesel':
      emission = ((0*vmean**2+0.2*vmean+16.4+(0/vmean))/(0*vmean**2+0.3*vmean+2.4)*(1-0))
      return emission 

    elif fuel_type == 'petrol':
      emission = ((0*vmean**2+0*vmean+2.6+(2.6/vmean))/(0*vmean**2+0*vmean+0.3)*(1-0))
      return emission

    elif fuel_type == 'lgp':
      emission = ((0*vmean**2+-0.3*vmean+27.6+(0/vmean))/(0*vmean**2+0*vmean+7.9)*(1-0))
      return emission

    else:
      print('It is not possible to calculate accurately your carbon emissions. Add at least the fuel type of your car.')
      emission_diesel = ((0*vmean**2+0.2*vmean+16.4+(0/vmean))/(0*vmean**2+0.3*vmean+2.4)*(1-0))
      emission_petrol = ((0*vmean**2+0*vmean+2.6+(2.6/vmean))/(0*vmean**2+0*vmean+0.3)*(1-0))
      emission_lpg = ((0*vmean**2+-0.3*vmean+27.6+(0/vmean))/(0*vmean**2+0*vmean+7.9)*(1-0))
      return emission_diesel, emission_lpg, emission_petrol
    


'''
    #selecting the values from the table
    if pollutant == 'CH4':
      slope == "Urban Off Peak"
      cur = conn.execute("SELECT DISTINCT alpha, beta, gamma, delta, epsilon, zita, hta, redu_factor from RoadEmission where Pollutant==? and Fuel==? and Euro_Standard==? and Segment==? and Road_Slope=? and Technology=?", (pollutant, fuel, standard, segment, slope, tech))
    else: 
'''    