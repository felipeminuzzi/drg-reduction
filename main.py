########################################################################################
# Functions for running DRG skeletal mechanism generation			       # 
# Author: Felipe C. Minuzzi							       #
# Date: 30/04/2018								       #
# Last Version: 18/04/2020                                                             #                                    
#										       #
########################################################################################


from __future__ import print_function
from __future__ import division
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import textwrap
from string import Template

def reactor(gas,nsp,fuel_species,tin0,pre0,tinf,npoints,npoints3,phi0,phif,npoints2,pref,data,datay):
      stoich_O2                       = gas.n_atoms(fuel_species,'C') + 0.25*gas.n_atoms(fuel_species,'H') \
                                        - 0.5*gas.n_atoms(fuel_species,'O')
      air_N2_O2_ratio                 = 3.76
      ifuel                           = gas.species_index(fuel_species)
      io2                             = gas.species_index('o2')
      in2                             = gas.species_index('n2')
      x                               = np.zeros(nsp,'d')
      x[io2]                          = stoich_O2
      x[in2]                          = stoich_O2*air_N2_O2_ratio
      gas.TPX                         = tin0,pre0,x
      r                               = ct.IdealGasReactor(contents=gas, name='Batch Reactor')
      reactorNetwork                  = ct.ReactorNet([r])
      stateVariableNames              = [r.component_name(item) for item in range(r.n_vars)]
      timeHistory                     = pd.DataFrame(columns=stateVariableNames)
########################################################################################
      def ignitionDelay(df, especie):
          """
          This function computes the ignition delay from the occurence of the
          peak in species' concentration.
          """
          return df[especie].idxmax()
########################################################################################

      Tmin                            = 1000./tinf
      Tmax                            = 1000./tin0 
      Ti2                             = np.zeros(npoints,'d')
      T                               = np.zeros(npoints,'d')
      ignition                        = np.zeros(npoints,'d')
      for j in range(npoints):
       Ti2[j]                         = Tmin + (Tmax - Tmin)*j/(npoints - 1)
       T[j]                           = 1000/Ti2[j]

      estimatedIgnitionDelayTimes     = np.ones(len(T))
      estimatedIgnitionDelayTimes[:6] = 6*[0.1]
      estimatedIgnitionDelayTimes[-2:]= 1
      estimatedIgnitionDelayTimes[-1] = 10
      ignitionDelays                  = pd.DataFrame(data={'T':T})
      ignitionDelays['ignDelay']      = np.nan

      for j in range(npoints3+1):
       phi                            = phi0 +((phif-phi0)/npoints3)*j
       x                              = np.zeros(nsp,'d')
       x[io2]                         = stoich_O2
       x[in2]                         = stoich_O2*air_N2_O2_ratio
       x[ifuel]                       = phi 
       for i in range(npoints2+1):
        if (i==0):
         pre                          =pre0*ct.one_atm
        else:
         pre                          = pre0*ct.one_atm + i*((pref-pre0)/npoints2)*ct.one_atm
        for i, temperature in enumerate(T):
         reactorTemperature           = temperature
         reactorPressure              = pre
         gas.TPX                      = reactorTemperature, reactorPressure, x
         r                            = ct.IdealGasReactor(contents=gas, name='Batch Reactor')
         reactorNetwork               = ct.ReactorNet([r])
         timeHistory                  = pd.DataFrame(columns=timeHistory.columns)
         t                            = 0
         counter                      = 0
         while t < estimatedIgnitionDelayTimes[i]:
             t = reactorNetwork.step()
             if not counter % 20:
                 timeHistory.loc[t] = r.get_state()
                 data.append(r.T)
                 datay.append(r.thermo.Y)
             counter += 1
         tau                          = ignitionDelay(timeHistory, 'OH')
         print('Computed Ignition Delay: {:.3e} s for T={}K, p={}atm, phi={}.'.format(tau, temperature, pre, phi))
      #  ignitionDelays.at(index=i, col='ignDelay', value=tau)
         ignition[i]                  = tau
      return data,datay

def flamespeed(gas,nsp,fuel_species,tin0,pre0,npoints3,phi0,phif):
      from pandas.core.common import flatten
      data2 = []
      datay2= []
      stoich_O2                       = gas.n_atoms(fuel_species,'C') + 0.25*gas.n_atoms(fuel_species,'H') \
                                        - 0.5*gas.n_atoms(fuel_species,'O')
      air_N2_O2_ratio                 = 3.76
      ifuel                           = gas.species_index(fuel_species)
      io2                             = gas.species_index('o2')
      in2                             = gas.species_index('n2')

      for j in range(npoints3):
       pre             = ct.one_atm
       phi             = phi0 +((phif-phi0)/npoints3)*j

       x               = np.zeros(nsp,'d')
       x[io2]          = stoich_O2
       x[in2]          = stoich_O2*air_N2_O2_ratio
       x[ifuel]        = phi
       gas.TPX         = tin0, pre, x
       initial_grid    = 2*np.array([0.0, 0.001, 0.01, 0.02, 0.029, 0.03],'d')
       f               = ct.FreeFlame(gas, initial_grid)
       f.inlet.X       = x
       f.inlet.T       = 298
       f.energy_enabled= False
       tol_ss          = [1.0e-5, 1.0e-9]
       tol_ts          = [1.0e-5, 1.0e-9]
       f.flame.set_steady_tolerances(default=tol_ss)
       f.flame.set_transient_tolerances(default=tol_ts)
       f.set_max_jac_age(50, 50)
       f.set_time_step(1.0e-5, [2, 5, 10, 20, 80])
       f.set_refine_criteria(ratio = 10.0, slope = 1, curve = 1)
       loglevel        = 1
       refine_grid     = False
       f.solve(loglevel, refine_grid)
       f.save('adiabatic.xml','no_energy','solution without the energy equation')
       f.energy_enabled= True
       f.set_refine_criteria(ratio = 10.0, slope = 1, curve = 1)
       f.solve(loglevel=1, refine_grid=True)
       f.save('adiabatic.xml','energy','solution with the energy equation enabled')
       f.energy_enabled= True
       f.set_refine_criteria(ratio = 9.0, slope = 0.9, curve = 0.9)
       f.solve(loglevel=1, refine_grid=True)
       f.save('adiabatic.xml','energy','solution with the energy equation enabled')
       f.energy_enabled= True
       f.set_refine_criteria(ratio = 7.0, slope = 0.7, curve = 0.7)
       f.solve(loglevel=1, refine_grid=True)
       f.save('adiabatic.xml','energy','solution with the energy equation enabled')
       f.energy_enabled= True
       f.set_refine_criteria(ratio = 5.0, slope = .5, curve = .5)
       f.solve(loglevel=1, refine_grid=True)
       f.save('adiabatic.xml','energy','solution with the energy equation enabled')
       f.energy_enabled= True
       f.set_refine_criteria(ratio = 3.75, slope = 0.375, curve = 0.375)
       f.solve(loglevel=1, refine_grid=True)
       f.save('adiabatic.xml','energy','solution with the energy equation enabled')
       f.energy_enabled= True
       f.set_refine_criteria(ratio = 3.355, slope = 0.355, curve = 0.355)
       f.solve(loglevel=1, refine_grid=True)
       f.save('adiabatic.xml','energy','solution with the energy equation enabled')
       temper  = list(f.T) 
       data2.append(temper)
       thermos = list(f.Y)
       datay2.append(thermos)
      data  = list(flatten(data2))
      datay = list(flatten(datay2))
      print(data)
      return data,datay

def coefficient(mechanism,gas,sn,sample,targets,species,nrx,species2,speciessk,epsilon,pre):
      for temp in range(sn):
       gas.TP   = sample[temp,0], pre
       gas.Y    = sample[temp,1:]
       fwd      = gas.forward_rate_constants
       bwd      = gas.reverse_rate_constants
       for a in targets:
        species.append(a)
        I = []
        deno     = np.zeros(nrx)
        denosum = 0
        I = [i for i,rea in enumerate(gas.reactions()) if a in rea.reactants or a in rea.products]
        for i in I:
         deno[i]  = fwd[i]-bwd[i]
        denosum  = sum(abs(deno)) 
        for cas in gas.species_names:
         II = []
         if cas != a:
          numesum = 0
          rab     = 0
          nume     = np.zeros(nrx)
          II = [i for i,rea in enumerate(gas.reactions()) if (a in rea.products or a in rea.reactants) and (cas in rea.reactants or cas in rea.products)]
          for i in II:
           nume[i]  = fwd[i]-bwd[i]
          numesum  = sum(abs(nume))      
          rab = numesum/denosum
          if II == []:
           rab=0
          if rab >= epsilon:
           species.append(cas)
      return species

def write_cti(solution,fuel_species,nsp1,epsilon):
    """Function to write cantera solution object to cti file.

    :param solution:
        Cantera solution object

    :return output_file_name:
        Name of trimmed mechanism file (.cti)

    >>> soln2cti.write(gas)
    """

    trimmed_solution = solution
    name             = fuel_species
    num              = str(nsp1)
    epsi             = str(epsilon)
    input_file_name_stripped = trimmed_solution.name
    cwd = os.getcwd()
    output_file_name = os.path.join(cwd,name + '_' +num+ 'sp'+'_threshold_'+epsi+'.cti')

    with open(output_file_name, 'w+') as f:

        #Get solution temperature and pressure
        solution_temperature = trimmed_solution.T
        solution_pressure = trimmed_solution.P

        #Work Functions

        # number of calories in 1000 Joules of energy
        calories_constant = 4184.0

        def eliminate(input_string, char_to_replace, spaces='single'):
            """
            Eliminate characters from a string

            :param input_string
                string to be modified
            :param char_to_replace
                array of character strings to be removed
            """
            for char in char_to_replace:
                input_string = input_string.replace(char, "")
            if spaces == 'double':
                input_string = input_string.replace(" ", "  ")
            return input_string

        def wrap_nasa(input_string):
            """
            Wrap string to cti NASA format width
            """
            output_string = textwrap.fill(
                                        input_string,
                                        width=50,
                                        subsequent_indent=16 * ' '
                                            )
            return output_string

        def section_break(title):
            """
            Insert break and new section title into cti file

            :param title:
                title string for next section_break
            """
            f.write('#' + '-' * 75 + '\n')
            f.write('#  ' + title + '\n')
            f.write('#' + '-' * 75 + '\n\n')

        def replace_multiple(input_string, replace_list):
            """
            Replace multiple characters in a string

            :param input_string
                string to be modified
            :param replace list
                list containing items to be replaced (value replaces key)
            """
            for original_character, new_character in replace_list.items():
                input_string = input_string.replace(original_character,
                                                    new_character)
            return input_string

        def build_arrhenius(equation_object, equation_type):
            """
            Builds Arrhenius coefficient string

            :param equation_objects
                cantera equation object
            :param equation_type:
                string of equation type
            """
            coeff_sum = sum(equation_object.reactants.values())
            pre_exponential_factor = equation_object.rate.pre_exponential_factor
            temperature_exponent = equation_object.rate.temperature_exponent
            activation_energy = (equation_object.rate.activation_energy /
                                calories_constant)

            if equation_type == 'ElementaryReaction':
                if coeff_sum == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))
                if coeff_sum == 2:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                if coeff_sum == 3:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))
            if equation_type == 'ThreeBodyReaction':
                if coeff_sum == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                if coeff_sum == 2:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))

            if (equation_type != 'ElementaryReaction'
                and equation_type != 'ThreeBodyReaction'):
                pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))

            arrhenius = [pre_exponential_factor,
                        temperature_exponent,
                        activation_energy
                        ]
            return str(arrhenius).replace("\'", "")

        def build_modified_arrhenius(equation_object, t_range):
            """
            Builds Arrhenius coefficient strings for high and low temperature ranges

            :param equation_objects
                cantera equation object
            :param t_range:
                simple string ('high' or 'low') to designate temperature range
            """

            if t_range == 'high':
                pre_exponential_factor = equation_object.high_rate.pre_exponential_factor
                temperature_exponent = equation_object.high_rate.temperature_exponent
                activation_energy = (equation_object.high_rate.activation_energy /
                                    calories_constant)
                if len(equation_object.products) == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                else:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))
                arrhenius_high = [pre_exponential_factor,
                                    temperature_exponent,
                                    activation_energy
                                    ]
                return str(arrhenius_high).replace("\'", "")

            if t_range == 'low':
                pre_exponential_factor = equation_object.low_rate.pre_exponential_factor
                temperature_exponent = equation_object.low_rate.temperature_exponent
                activation_energy = (equation_object.low_rate.activation_energy /
                                    calories_constant)

                if len(equation_object.products) == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))
                else:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                arrhenius_low = [pre_exponential_factor,
                                temperature_exponent,
                                activation_energy
                                ]
                return str(arrhenius_low).replace("\'", "")

        def build_falloff(j):
            """
            Creates falloff reaction Troe parameter string

            param j:
                Cantera falloff parameters object
            """
            falloff_string = str(
                        ',\n        falloff = Troe(' +
                        'A = ' + str(j[0]) +
                        ', T3 = ' + str(j[1]) +
                        ', T1 = ' + str(j[2]) +
                        ', T2 = ' + str(j[3]) + ')       )\n\n'
                        )
            return falloff_string

        def build_species_string():
            """
            Formats species list at top of mechanism file
            """
            species_list_string = ''
            line = 1
            for sp_str in trimmed_solution.species_names:
                #get length of string next species is added
                length_new = len(sp_str)
                length_string = len(species_list_string)
                total = length_new +length_string +3
                #if string will go over width, wrap to new line
                if line == 1:
                    if total >= 55:
                        species_list_string += '\n'
                        species_list_string += ' ' * 17
                        line += 1
                if line > 1:
                    if total >= 70 * line:
                        species_list_string += '\n'
                        species_list_string += ' ' * 17
                        line += 1
                species_list_string += sp_str + ' '
            return species_list_string

        #Write title block to file

        section_break('CTI File converted from Solution Object')
        unit_string = "units(length = \"cm\", time = \"s\"," +\
                            " quantity = \"mol\", act_energy = \"cal/mol\")"
        f.write(unit_string + '\n\n')

        #Write Phase definition to file

        element_names = eliminate(str(trimmed_solution.element_names),
                                  ['[',
                                  ']',
                                  '\'',
                                  ','])
        element_names = element_names.replace('AR', 'Ar')
        species_names = build_species_string()
        phase_string = Template(
                        'ideal_gas(name = \"$input_file_name_stripped\", \n' +
                        '     elements = \"$elements\", \n' +
                        '     species =""" $species""", \n' +
                        '     reactions = \"all\", \n' +
                        '     initial_state = state(temperature = $solution_temperature, '
                        'pressure= $solution_pressure)   )       \n\n'
                       )

        f.write(phase_string.substitute(
                                elements=element_names,
                                species=species_names,
                                input_file_name_stripped=input_file_name_stripped,
                                solution_temperature=solution_temperature,
                                solution_pressure=solution_pressure
                                ))

        #Write species data to file

        section_break('Species data')
        for sp_index, name in enumerate(trimmed_solution.species_names):
            #joules/kelvin, boltzmann constant
            boltzmann = ct.boltzmann
            #1 debye = d coulomb-meters
            debeye_conversion_constant = 3.33564e-30
            species = trimmed_solution.species(sp_index)
            name = str(trimmed_solution.species(sp_index).name)
            nasa_coeffs = trimmed_solution.species(sp_index).thermo.coeffs
            replace_list_1 = {'{':'\"',
                              '}':'\"',
                              '\'':'',
                              ':  ':':',
                              '.0':"",
                              ',':'',
                              ' ': '  '
                              }
            #build 7-coeff NASA polynomial array
            nasa_coeffs_1 = []
            for j, k in enumerate(nasa_coeffs):
                coeff = "{:.9e}".format(nasa_coeffs[j+8])
                nasa_coeffs_1.append(coeff)
                if j == 6:
                    nasa_coeffs_1 = wrap_nasa(eliminate(str(nasa_coeffs_1),
                                                        {'\'':""}))
                    break
            nasa_coeffs_2 = []
            for j, k in enumerate(nasa_coeffs):
                coeff = "{:.9e}".format(nasa_coeffs[j+1])
                nasa_coeffs_2.append(coeff)
                if j == 6:
                    nasa_coeffs_2 = wrap_nasa(eliminate(
                                                str(nasa_coeffs_2),
                                                {'\'':""}))
                    break
            #Species attributes from trimmed solution object
            composition = replace_multiple(
                                            str(species.composition),
                                                replace_list_1)
            nasa_range_1 = str([species.thermo.min_temp, nasa_coeffs[0]])
            nasa_range_2 = str([nasa_coeffs[0], species.thermo.max_temp])
            #check if species has defined transport data
            if bool(species.transport) is True:
                transport_geometry = species.transport.geometry
                diameter = str(species.transport.diameter*(10**10))
                well_depth = str(species.transport.well_depth/boltzmann)
                polar = str(species.transport.polarizability*10**30)
                rot_relax = str(species.transport.rotational_relaxation)
                dipole = str(species.transport.dipole/debeye_conversion_constant)
                #create and fill string templates for each species
                if species.transport.dipole != 0:
                    species_string = Template(
                            'species(name = "$name",\n' +
                            '    atoms = $composition, \n' +
                            '    thermo = (\n' +
                            '       NASA(   $nasa_range_1, $nasa_coeffs_1  ),\n' +
                            '       NASA(   $nasa_range_2, $nasa_coeffs_2  )\n' +
                            '               ),\n'
                            '    transport = gas_transport(\n' +
                            '                   geom = \"$transport_geometry\",\n' +
                            '                   diam = $diameter, \n' +
                            '                   well_depth = $well_depth, \n' +
                            '                   polar = $polar, \n' +
                            '                   rot_relax = $rot_relax, \n' +
                            '                   dipole= $dipole) \n' +
                            '        )\n\n'
                            )
                    f.write(species_string.substitute(
                                name=name,
                                composition=composition,
                                nasa_range_1=nasa_range_1,
                                nasa_coeffs_1=nasa_coeffs_1,
                                nasa_range_2=nasa_range_2,
                                nasa_coeffs_2=nasa_coeffs_2,
                                transport_geometry=transport_geometry,
                                diameter=diameter,
                                well_depth=well_depth,
                                polar=polar,
                                rot_relax=rot_relax,
                                dipole=dipole
                                ))
                if species.transport.dipole == 0:
                    species_string = Template(
                            'species(name = "$name",\n'
                            '    atoms = $composition, \n'
                            '    thermo = (\n'
                            '       NASA(   $nasa_range_1, $nasa_coeffs_1  ),\n'
                            '       NASA(   $nasa_range_2, $nasa_coeffs_2  )\n'
                            '               ),\n'
                            '    transport = gas_transport(\n'
                            '                   geom = \"$transport_geometry\",\n'
                            '                   diam = $diameter, \n'
                            '                   well_depth = $well_depth, \n'
                            '                   polar = $polar, \n'
                            '                   rot_relax = $rot_relax) \n'
                            '        )\n\n'
                            )
                    f.write(species_string.substitute(
                                name=name,
                                composition=composition,
                                nasa_range_1=nasa_range_1,
                                nasa_coeffs_1=nasa_coeffs_1,
                                nasa_range_2=nasa_range_2,
                                nasa_coeffs_2=nasa_coeffs_2,
                                transport_geometry=transport_geometry,
                                diameter=diameter,
                                well_depth=well_depth,
                                polar=polar,
                                rot_relax=rot_relax,
                                ))
            if bool(species.transport) is False:
                species_string = Template(
                            'species(name = "$name",\n'
                            '    atoms = $composition, \n'
                            '    thermo = (\n'
                            '       NASA(   $nasa_range_1, $nasa_coeffs_1  ),\n'
                            '       NASA(   $nasa_range_2, $nasa_coeffs_2  )\n'
                            '               ),\n'
                            '        )\n\n'
                            )
                f.write(species_string.substitute(
                                name=name,
                                composition=composition,
                                nasa_range_1=nasa_range_1,
                                nasa_coeffs_1=nasa_coeffs_1,
                                nasa_range_2=nasa_range_2,
                                nasa_coeffs_2=nasa_coeffs_2,
                                ))

        #Write reactions to file

        section_break('Reaction Data')

        #write data for each reaction in the Solution Object
        for eq_index in range(len(trimmed_solution.reaction_equations())):
            equation_string = str(trimmed_solution.reaction_equation(eq_index))
            equation_object = trimmed_solution.reaction(eq_index)
            equation_type = type(equation_object).__name__
            m = str(eq_index+1)
            if equation_type == 'ThreeBodyReaction':
                #trimms efficiencies list
                efficiencies = equation_object.efficiencies
                trimmed_efficiencies = equation_object.efficiencies
                for s in efficiencies:
                    if s not in trimmed_solution.species_names:
                        del trimmed_efficiencies[s]
                arrhenius = build_arrhenius(equation_object, equation_type)
                replace_list_2 = {"{":  "\"",
                                  "\'": "",
                                  ": ": ":",
                                  ",":  " ",
                                  "}":  "\""
                                  }
                efficiencies_string = replace_multiple(
                                                    str(trimmed_efficiencies),
                                                        replace_list_2)
                reaction_string = Template(
                            '#  Reaction $m\n'
                            'three_body_reaction( \"$equation_string\",  $Arr,\n'
                            '       efficiencies = $Efficiencies) \n\n'
                            )
                f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        Arr=arrhenius,
                        Efficiencies=efficiencies_string
                        ))
            if equation_type == 'ElementaryReaction':
                arrhenius = build_arrhenius(equation_object, equation_type)
                if equation_object.duplicate is True:
                    reaction_string = Template(
                            '#  Reaction $m\n'
                            'reaction( \"$equation_string\", $Arr,\n'
                            '        options = \'duplicate\')\n\n'
                            )
                else:
                    reaction_string = Template(
                            '#  Reaction $m\n'
                            'reaction( \"$equation_string\", $Arr)\n\n'
                            )
                f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        Arr=arrhenius
                        ))
            if equation_type == 'FalloffReaction':
                #trimms efficiencies list
                efficiencies = equation_object.efficiencies
                trimmed_efficiencies = equation_object.efficiencies
                for s in efficiencies:
                    if s not in trimmed_solution.species_names:
                        del trimmed_efficiencies[s]

                kf = build_modified_arrhenius(equation_object, 'high')
                kf0 = build_modified_arrhenius(equation_object, 'low')
                replace_list_2 = {
                                "{":"\"",
                                "\'":"",
                                ": ":":",
                                ",":" ",
                                "}":"\""
                                }
                efficiencies_string = replace_multiple(
                                                str(trimmed_efficiencies),
                                                    replace_list_2)
                reaction_string = Template(
                            '#  Reaction $m\n' +
                            'falloff_reaction( \"$equation_string\",\n' +
                            '        kf = $kf,\n' +
                            '        kf0   = $kf0,\n' +
                            '        efficiencies = $Efficiencies'
                            )
                f.write(reaction_string.substitute(
                        m=m,
                        equation_string=equation_string,
                        kf=kf,
                        kf0=kf0,
                        Efficiencies=efficiencies_string
                        ))
                j = equation_object.falloff.parameters
                #If optional Arrhenius data included:
                try:
                    falloff_str = build_falloff(j)
                    f.write(falloff_str)
                except IndexError:
                    f.write('\n           )\n\n')
    return output_file_name

def write_inp(solution,fuel_species,nsp1,epsilon):
    """Function to write cantera solution object to inp file.

    :param solution:
        Cantera solution object

    :return:
        Name of trimmed Mechanism file (.inp)

    >>> soln2ck.write(gas)
    """
    trimmed_solution = solution
    name             = fuel_species
    num              = str(nsp1)
    epsi             = str(epsilon)
    input_file_name_stripped = trimmed_solution.name
    cwd = os.getcwd()
    output_file_name = os.path.join(cwd,name + '_' +num+ 'sp'+'_threshold_'+epsi+'.inp')
    with open(output_file_name, 'w+') as f:

        #Work functions

        calories_constant = 4184.0 #number of calories in 1000 Joules of energy
        def eliminate(input_string, char_to_replace, spaces='single'):
            """
            Eliminate characters from a string

            :param input_string
                string to be modified
            :param char_to_replace
                array of character strings to be removed
            """
            for char in char_to_replace:
                input_string = input_string.replace(char, "")
            if spaces == 'double':
                input_string = input_string.replace(" ", "  ")
            return input_string


        def replace_multiple(input_string, replace_list):
            """
            Replace multiple characters in a string

            :param input_string
                string to be modified
            :param replace list
                list containing items to be replaced (value replaces key)
            """
            for original_character, new_character in replace_list.items():
                input_string = input_string.replace(original_character,
                                                    new_character)
            return input_string

        def build_arrhenius(equation_object, equation_type):
            """
            Builds Arrhenius coefficient string

            :param equation_objects
                cantera equation object
            :param equation_type:
                string of equation type
            """
            coeff_sum = sum(equation_object.reactants.values())
            pre_exponential_factor = equation_object.rate.pre_exponential_factor
            temperature_exponent = '{:.3f}'.format(equation_object.rate.temperature_exponent)
            activation_energy = '{:.2f}'.format(equation_object.rate.activation_energy / calories_constant)
            if equation_type == 'ElementaryReaction':
                if coeff_sum == 1:
                    pre_exponential_factor = str(
                                    '{:.3E}'.format(pre_exponential_factor))
                if coeff_sum == 2:
                    pre_exponential_factor = str(
                                    '{:.3E}'.format(pre_exponential_factor*10**3))
                if coeff_sum == 3:
                    pre_exponential_factor = str(
                                    '{:.3E}'.format(pre_exponential_factor*10**6))
            if equation_type == 'ThreeBodyReaction':
                if coeff_sum == 1:
                    pre_exponential_factor = str(
                                    '{:.3E}'.format(pre_exponential_factor*10**3))
                if coeff_sum == 2:
                    pre_exponential_factor = str(
                                    '{:.3E}'.format(pre_exponential_factor*10**6))
            if (equation_type != 'ElementaryReaction'
                and equation_type != 'ThreeBodyReaction'):
                pre_exponential_factor = str(
                                    '{:.3E}'.format(pre_exponential_factor))
            arrhenius = [pre_exponential_factor,
                    temperature_exponent,
                    activation_energy]
            return arrhenius

        def build_modified_arrhenius(equation_object, t_range):
            """
            Builds Arrhenius coefficient strings for high and low temperature ranges

            :param equation_objects
                cantera equation object
            :param t_range:
                simple string ('high' or 'low') to designate temperature range
            """
            if t_range == 'high':
                pre_exponential_factor = equation_object.high_rate.pre_exponential_factor
                temperature_exponent = '{:.3f}'.format(equation_object.high_rate.temperature_exponent)
                activation_energy = '{:.2f}'.format(equation_object.high_rate.activation_energy/calories_constant)
                if len(equation_object.products) == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))
                else:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor))
                arrhenius_high = [pre_exponential_factor,
                            temperature_exponent,
                            activation_energy]
                return arrhenius_high
            if t_range == 'low':

                pre_exponential_factor = equation_object.low_rate.pre_exponential_factor
                temperature_exponent = '{:.3f}'.format(equation_object.low_rate.temperature_exponent)
                activation_energy = '{:.2f}'.format(equation_object.low_rate.activation_energy/calories_constant)
                if len(equation_object.products) == 1:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**6))
                else:
                    pre_exponential_factor = str(
                                    '{:.5E}'.format(pre_exponential_factor*10**3))

                arrhenius_low = [pre_exponential_factor,
                            temperature_exponent,
                            activation_energy]
                return arrhenius_low


        def build_nasa(nasa_coeffs, row):
            """
            Creates string of nasa polynomial coefficients

            :param nasa_coeffs
                cantera species thermo coefficients object
            :param row
                which row to write coefficients in
            """
            line_coeffs = ''
            lines = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14]]
            line_index = lines[row-2]
            for ix, c in enumerate(nasa_coeffs):
                if ix in line_index:
                    if c >= 0:
                        line_coeffs += ' '
                    line_coeffs += str('{:.8e}'.format(c))
            return line_coeffs

        def build_species_string():
            """
            formats species string for writing
            """
            species_list_string = ''
            line = 1
            for sp_index, sp_string in enumerate(trimmed_solution.species_names):
                sp = ' '
                #get length of string next species is added
                length_new = len(sp_string)
                length_string = len(species_list_string)
                total = length_new + length_string + 3
                #if string will go over width, wrap to new line
                if total >= 70*line:
                    species_list_string += '\n'
                    line += 1
                species_list_string += sp_string + ((16-len(sp_string))*sp)
            return species_list_string

        title = ''
        section_break = ('!'+ "-"*75 + '\n'
                         '!  ' + title +'\n'
                         '!'+ "-"*75 + '\n')

        #Write title block to file
        title = 'Chemkin File converted from Solution Object by pyMARS'
        f.write(section_break)

        #Write phase definition to file
        element_names = eliminate(str(trimmed_solution.element_names),
                                            ['[', ']', '\'', ','])
        element_string = Template(
                                'ELEMENTS\n' +
                                '$element_names\n' +
                                'END\n')
        f.write(element_string.substitute(element_names=element_names))
        species_names = build_species_string()
        species_string = Template(
                                'SPECIES\n' +
                                '$species_names\n'+
                                'END\n')
        f.write(species_string.substitute(species_names=species_names))

        #Write species to file
        title = 'Species data'
        f.write(section_break)

        f.write('THERMO ALL' + '\n' +
                '   300.000  1000.000  5000.000' +'\n')
        phase_unknown_list = []

        #write data for each species in the Solution object
        for sp_index in range(len(trimmed_solution.species_names)):
            d = 3.33564e-30 #1 debye = d coulomb-meters
            species = trimmed_solution.species(sp_index)
            name = str(trimmed_solution.species(sp_index).name)
            nasa_coeffs = trimmed_solution.species(sp_index).thermo.coeffs
            #Species attributes from trimmed solution object
            t_low = '{0:.3f}'.format(species.thermo.min_temp)
            t_max = '{0:.3f}'.format(species.thermo.max_temp)
            t_mid = '{0:.3f}'.format(species.thermo.coeffs[0])
            temp_range = str(t_low) + '  ' + str(t_max) + '  ' + t_mid
            species_comp = ''
            for atom in species.composition:
                species_comp += '{:<4}'.format(atom)
                species_comp += str(int(species.composition[atom]))
            if type(species.transport).__name__ == 'GasTransportData':
                species_phase = 'G'
            else:
                phase_unknown_list.append(name)
                species_phase = 'G'
            line_1 = (
                    '{:<18}'.format(name) +
                    '{:<6}'.format('    ') +
                    '{:<20}'.format(species_comp) +
                    '{:<4}'.format(species_phase) +
                    '{:<31}'.format(temp_range) +
                    '{:<1}'.format('1') +
                    '\n')
            f.write(line_1)
            line_2_coeffs = build_nasa(nasa_coeffs, 2)
            line_2 = line_2_coeffs  + '    2\n'
            f.write(line_2)
            line_3_coeffs = build_nasa(nasa_coeffs, 3)
            line_3 = line_3_coeffs + '    3\n'
            f.write(line_3)
            line_4_coeffs = build_nasa(nasa_coeffs, 4)
            line_4 = line_4_coeffs + '                   4\n'
            f.write(line_4)

        f.write('END\n')

        #Write reactions to file
        title = 'Reaction Data'
        f.write(section_break)
        f.write('REACTIONS\n')
        #write data for each reaction in the Solution Object
        for reac_index in range(len(trimmed_solution.reaction_equations())):
            equation_string = str(trimmed_solution.reaction_equation(reac_index))
            equation_string = eliminate(equation_string, ' ', 'single')
            equation_object = trimmed_solution.reaction(reac_index)
            equation_type = type(equation_object).__name__
            m = str(reac_index+1)
            if equation_type == 'ThreeBodyReaction':
                arrhenius = build_arrhenius(equation_object, equation_type)
                main_line = (
                            '{:<51}'.format(equation_string) +
                            '{:>9}'.format(arrhenius[0]) +
                            '{:>9}'.format(arrhenius[1]) +
                            '{:>11}'.format(arrhenius[2]) +
                            '\n')
                f.write(main_line)
                #trimms efficiencies list
                efficiencies = equation_object.efficiencies
                trimmed_efficiencies = equation_object.efficiencies
                for s in efficiencies:
                    if s not in trimmed_solution.species_names:
                        del trimmed_efficiencies[s]
                replace_list_2 = {
                                    '{':'',
                                    '}':'/',
                                    '\'':'',
                                    ':':'/',
                                    ',':'/'}
                efficiencies_string = replace_multiple(
                                            str(trimmed_efficiencies),
                                                replace_list_2)
                secondary_line = str(efficiencies_string) + '\n'
                if bool(efficiencies) is True:
                    f.write(secondary_line)
            if equation_type == 'ElementaryReaction':
                arrhenius = build_arrhenius(equation_object, equation_type)
                main_line = (
                            '{:<51}'.format(equation_string) +
                            '{:>9}'.format(arrhenius[0]) +
                            '{:>9}'.format(arrhenius[1]) +
                            '{:>11}'.format(arrhenius[2]) +
                            '\n')
                f.write(main_line)
            if equation_type == 'FalloffReaction':
                arr_high = build_modified_arrhenius(equation_object, 'high')
                main_line = (
                            '{:<51}'.format(equation_string) +
                            '{:>9}'.format(arr_high[0]) +
                            '{:>9}'.format(arr_high[1]) +
                            '{:>11}'.format(arr_high[2]) +
                            '\n')
                f.write(main_line)
                arr_low = build_modified_arrhenius(equation_object, 'low')
                second_line = (
                                '     LOW  /' +
                                '  ' + arr_low[0] +
                                '  ' + arr_low[1] +
                                '  ' + arr_low[2] + '/\n')
                f.write(second_line)
                j = equation_object.falloff.parameters
                #If optional Arrhenius data included:
                try:
                    third_line = (
                                '     TROE/' +
                                '   ' + str(j[0]) +
                                '  ' + str(j[1]) +
                                '  ' + str(j[2]) +
                                '  ' + str(j[3]) +' /\n')
                    f.write(third_line)
                except IndexError:
                    pass
                #trimms efficiencies list
                efficiencies = equation_object.efficiencies
                trimmed_efficiencies = equation_object.efficiencies
                for s in efficiencies:
                    if s not in trimmed_solution.species_names:
                        del trimmed_efficiencies[s]
                replace_list_2 = {
                                    '{':'',
                                    '}':'/',
                                    '\'':'',
                                    ':':'/',
                                    ',':'/'}
                efficiencies_string = replace_multiple(
                                            str(trimmed_efficiencies),
                                                replace_list_2)

                fourth_line = str(efficiencies_string) + '\n'
                if bool(efficiencies) is True:
                    f.write(fourth_line)
            #dupluicate option
            if equation_object.duplicate is True:
                duplicate_line = ' DUPLICATE' +'\n'
                f.write(duplicate_line)
        f.write('END')
    return output_file_name
