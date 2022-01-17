class EuroStandards:

  def find_standard(year, segment=None):

    standard = None
    if segment=='Passenger Cars':
      if year > 1992 and year <=1996:
        standard = 'Euro 1'
      elif year > 1996 and year <=2000:
        standard = 'Euro 2'
      elif year > 2000 and year <=2005:
        standard = 'Euro 3'
      elif year > 2005 and year <=2009:
        standard = 'Euro 4'
      elif year > 2009 and year <=2014:
        standard = 'Euro 5'
      elif year > 2014:
        standard = 'Euro 6 a/b/c'

    if segment=='Light Commercial Vehicle':
      if year>=1994 and year <=1998:
        standard='Euro 1'
      elif year>=1998 and year <=2000:
        standard = 'Euro 2'
      elif year>=2000 and year <=2005:
        standard = 'Euro 3'
      elif year>=2005 and year <=2010:
        standard = 'Euro 4'
      elif year>=2010 and year <=2015:
        standard = 'Euro 5'
      elif year>=2015:
        standard = 'Euro 6 a/b/c'

    if segment=='Heavy Duty Trucks':
      if year>=1992 and year <=1995:
        standard='Euro 1'
      elif year>=1995 and year <=1999:
        standard = 'Euro 2'
      elif year>=1999 and year <=2005:
        standard = 'Euro 3'
      elif year>=2005 and year <=2008:
        standard = 'Euro 4'
      elif year>=2008 and year <=2013:
        standard = 'Euro 5'
      elif year>=2013:
        standard = 'Euro 6 a/b/c'
    
    if segment=='motorcycle':
      if year>=2000 and year <=2004:
        standard='Euro 1'
      elif year>=2004 and year <=2007:
        standard = 'Euro 2'
      elif year>=2007:
        standard = 'Euro 3'

    return standard