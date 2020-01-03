"""
Домашнее задание
Написать классы по аналогии с Rectangle для вичисления площади и обьема следующих фигур: 
    а) Квадрат 
    б) Круг (с использованием полярных координат) 
    в) параллелограм 
    г) Трапеция 
    д) Треугольник 
    е) Ромб
Используя написанные классы вычислить значения площади 10 произвольных фигур 
Разделяющую поверхность в двумерном пространстве задана формулой: 
    10 * x1 - 2 * x2 + 3 = 0 Случайным образом 
    на 2- мерную поверхность набросаны точки 
    x1x2 = np.random.randn(200, 2) 
    Отобразить точки и разделяющую поверхность на графике, покрасить точки выше разделяющей поверхности в синий, 
    а ниже в красный цвета. 
Для задания 3 рассчитайте среднее расстояние от точек до разделяющей поверхности и отобразите только те точки, 
    расстояние до которых больше среднего

PRO:
Написать функцию, для семплирования 1000 точек из 3-х мерной сферы с координатами цветового пространства 
    LAB (https://en.wikipedia.org/wiki/CIELAB_color_space).
Визуализировать результат в 3-х мерной проекции .
Даны точки в двумерном пространстве 
    import numpy as np 
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Построить разделяющую поверхность такую, чтобы отделить друг от друга 
    первые 20 и последние 20 точек из X ( [0] * 20 + [1] * 20 )
Нарисовать график.
"""

import numpy as np
import seaborn
import matplotlib.pyplot as plt
from   mpl_toolkits.mplot3d import Axes3D


#from mpl_toolkits.mplot3d import Axes3D
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

class CalcArea:
    def __init__(self, *args):
        CalcArea.calc(self, *args)
        
    def calc(self, *args):
        count_args = len(args)

        if count_args <= 1:
            return 0

        self.type_area = args[0]

        if  self.type_area == "Square" and len(args)>=2:        # Квадрат
                return args[1]**2  # S = a**2  

        if  (self.type_area  == "Rect" or self.type_area  == "Parallelogram") and len(args)>=3:  # Прямоугольник or Параллелограмм
            return  args[1]*args[2]  # S = a*b  
                
        if  self.type_area  == "Circle" and len(args)>=2:       # Круга
            return  np.pi * args[1] **2 
                
        if  self.type_area  == "Trapezioid" and len(args)>=4:   # Trapezioid # S = (a+ b) /(2) * h
            return  (args[1] + args[2]) /2. * float(args[3])
                
        if  self.type_area  == "Triangle" and len(args)>=4:     # Triangle  S = 1/2 * a* b * sin(gamma)
            return  args[1] * args[2] * np.sin(args[3]) / 2.
                
        if  self.type_area  == "Rhombus" and len(args)>=3:      # S =  a**2 * sin(alpha)
            return  args[1] **2 * np.sin(args[2])
        
        return 0                

class MySection:
    def __init__(self ):
        pass

    def line(self, x1, x2):     # Разделяющая поверхность: w1 * x1 + w2 * x2 + b = 0    #surface(self):
        return 10 * x1 - 2 * x2 + 3                

    def line_x1(self, x1):      # служебная функция 
        return (10 * x1 + 3) / 2

    def set_generat(self, n):# генерируем случайные точки
        self.n_gen = n
        np.random.seed(0)
        self.x1x2 = np.random.randn(self.n_gen, 2) *2

    def plot_gen_sign(self, *args): # рисуем сгенерированные точки
        filtr0=0.0
        if len(args)>=1:
                filtr0=args[0]

        for x1, x2 in self.x1x2:
            value = MySection.line(self, x1, x2)

            if filtr0==0.0:            
                if (value > 0): color_ = 'blue'
                elif (value < 0):  color_ = 'red'
                else: color_ = 'green'
                plt.plot(x1, x2, 'ro', color = color_) 
            else:
                if np.abs(value) >=filtr0:
                    color_ = 'blue' if (value > 0) else  'red'
                    plt.plot(x1, x2, 'ro', color = color_) 
        # нормализация осей
        plt.gca().set_aspect('equal', adjustable='box')        
            
        # рисуем разделяющую поверхность
        x1_range = np.arange(-1.5, 1.5, 0.5)
        plt.plot(x1_range, MySection.line_x1(self, x1_range), color='black')

        # Подписываем оси
        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.show()

    def average_distance(self):  # Вычислим среднее расстояние по всем точкам:
        ls_dist = []
        for x1, x2 in self.x1x2:
            ls_dist.append(np.abs(MySection.line(self, x1, x2))) # Берем абсолютное значение расстояния, так как нам не важно слева точка от разделяющей поверхности или справа
        return ls_dist

class MySection2:
    def __init__(self ):
        self.labels = []
        self.x1x2 = {}
        self.w1_, self.w2_, self.b_, self.lr =0.0, 0.0, 0.0, 0.1
        self.lines=[]

    def set_generat(self, n):# генерируем случайные точки
        self.n_gen = n
        np.random.seed(0)
        self.x1x2 =np.r_[np.random.randn(self.n_gen, 2) - [2, 2], np.random.randn(self.n_gen, 2) + [2, 2]]

    def plot_x1x2(self):
        for x1, x2 in self.x1x2:
            plt.plot(x1, x2, 'ro', color='blue')
        # нормализация осей
        plt.gca().set_aspect('equal', adjustable='box')

        # Подписываем оси
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()        

    def plot_x1x2_1(self):
        for  (x1, x2) in enumerate(self.x1x2):
            pred_label = MySection2.decision_unit(self, MySection2.lr_line(self, x1, x2))

            if (pred_label < 0):
                plt.plot(x1, x2, 'ro', color='green')
            else:
                plt.plot(x1, x2, 'ro', color='red')

        # выставляем равное пиксельное разрешение по осям
        plt.gca().set_aspect('equal', adjustable='box')    

        # проставляем названия осей
        plt.xlabel('x1')
        plt.ylabel('x2')

        # служебный диапазон для визуализации границы решений
        x1_range = np.arange(-5, 5, 0.1)

        # функционал, возвращающий границу решений в пригодном для отрисовки виде
        # x2 = f(x1) = -(w1 * x1 + b) / w2
        def f_lr_line(w1, w2, b):
            def lr_line(x1):
                return -(w1 * x1 + b) / w2
            return lr_line

        # отрисовываем историю изменения границы решений
        it = 0
        for coeff in self.lines:
            lr_line = f_lr_line(coeff[0], coeff[1], coeff[2])
            
            plt.plot(x1_range, lr_line(x1_range), label = 'it: ' + str(it))
            
            it = it + 1
            
        # зум
        plt.axis([-5, 5, -5, 5])
            
        # легенда
        plt.legend(loc = 'lower left')
        
        # на экран!
        plt.show()

    def to_paint_point(self):
        self.labels = []        # рисуем сгенерированные точки
        for idx, x in enumerate(self.x1x2):
            x1 = x[0]
            x2 = x[1]

            if idx <self.n_gen:
                plt.plot(x1, x2, 'ro', color='green') # Первые n точек подряд красим в зеленый цвет (1-й класс) и назначим метки классов: - 1
                self.labels.append(-1)
            elif idx >=self.n_gen:
                plt.plot(x1, x2, 'ro', color='red')  # Последние n точек подряд красим в красный цвет (2-й класс) и назначим метки классов : + 1
                self.labels.append(1)

        # нормализация осей
        plt.gca().set_aspect('equal', adjustable='box')
                    
        # Подписываем оси
        plt.xlabel('x1')
        plt.ylabel('x2')

        plt.show()

    def lr_line(self, x1, x2):
        return self.w1_ * x1 + self.w2_ * x2 + self.b_

    def decision_unit(self, value):
        return -1 if value < 0 else 1

    def func_inicial(self):
        self.labels = np.array(self.labels)
        indices = np.array(range(self.x1x2.shape[0]))
        np.random.shuffle(indices)
        self.x1x2 = self.x1x2[indices]
        self.labels = self.labels[indices]
        self.w1_ = np.random.uniform(-2,2) 
        self.w2_ = np.random.uniform(-2,2)
        self.b_ = np.random.uniform(-30,30)

    def building_surface(self): # строим поверхность
        self.lines = [[self.w1_, self.w2_, self.b_]]    # добавляем начальное разбиение в список

        for max_iter in range(100):
            # счётчик неверно классифицированных примеров
            # для ранней остановки
            mismatch_count = 0
            
            # по всем образцам
            for i, (x1, x2) in enumerate(self.x1x2):
                # считаем значение линейной комбинации на гиперплоскости
                value = MySection2.lr_line(self,x1, x2)
                
                # класс из тренировочного набора (-1, +1)
                true_label = int(self.labels[i])
                
                # предсказанный класс (-1, +1)
                pred_label = MySection2.decision_unit(self, value)
                
                # если имеет место ошибка классификации
                if (true_label != pred_label):
                    # корректируем веса в сторону верного класса, т.е.
                    # идём по нормали — (x1, x2) — в случае класса +1
                    # или против нормали — (-x1, -x2) — в случае класса -1
                    # т.к. нормаль всегда указывает в сторону +1
                    self.w1_ = self.w1_ + x1 * true_label * self.lr
                    self.w2_ = self.w2_ + x2 * true_label * self.lr
                    
                    # смещение корректируется по схожему принципу
                    self.b_ = self.b_ + true_label
                    
                    # считаем количество неверно классифицированных примеров
                    mismatch_count = mismatch_count + 1
            
            # если была хотя бы одна коррекция
            if (mismatch_count > 0):
                # запоминаем границу решений
                self.lines.append([self.w1_, self.w2_, self.b_])
            else:
                # иначе — ранняя остановка
                break

class Sphere():
    def __init__(self, x0, y0, z0):
        self.x0, self.y0, self.z0 = x0, y0, z0

    def points_in_poligon(self, n):
      points=[]      
      randpi =lambda x : np.random.uniform(0, x*np.pi)
      r_l = np.random.uniform(0,128, n)
      r_a = np.random.uniform(-128,128, n)
      r_b = np.random.uniform(-128,128, n)

      for i in range(n):
          alfa=randpi(2)
          beta=randpi(1)
          points.append([  
              self.x0 + r_l[i] * np.sin(alfa)*np.cos(beta), 
              self.y0 + r_a[i] * np.sin(alfa)*np.sin(beta), 
              self.z0 + r_b[i] * np.cos(alfa)])

      ii=len(points)                
      return points

def L_2():
    calcaraa_ = CalcArea()
#     тест функумй  площади  
    randInt=lambda x0, x1:np.random.randint(x0, x1)
    randpi =lambda x : np.random.uniform(0, x*np.pi)

    print(" Площадь квадрата {} ".format(calcaraa_.calc("Square", randInt(1, 200)))) 
    print(" Площадь прямоугольника {} ".format(calcaraa_.calc("Rect", randInt(1, 200), randInt(10, 300)))) 
    print(" Площадь параллелограмм {} ".format(calcaraa_.calc("Parallelogram", randInt(1, 200), randInt(10, 300)))) 
    print(" Площадь круга {} ".format(calcaraa_.calc("Circle", randInt(1, 200)))) 
    print(" Площадь трапеция {} ".format(calcaraa_.calc("Trapezioid", randInt(1, 200), randInt(10, 300),randInt(1, 200)))) 
    print(" Площадь Триугольник {} ".format(calcaraa_.calc("Triangle", randInt(1, 200), randInt(10, 300), randpi(1/6.)))) 
    print(" Площадь Ромб {} ".format(calcaraa_.calc("Rhombus", randInt(1, 200), randpi(1/6.)))) 
    
def L_3():
    my_section = MySection()
    my_section.set_generat(1000)
    my_section.plot_gen_sign()
    ls= my_section.average_distance()
    porog=sum(ls)/float(len(ls))

    print("Среднее расстояние по всем точкам до разделяющей поверхности равно: {}".format(porog))
    my_section.plot_gen_sign(porog)

def L_4():
    my_section = MySection2()
    my_section.set_generat(200)
    my_section.plot_x1x2()
    my_section.to_paint_point()
    my_section.func_inicial()
    my_section.building_surface()
    my_section.plot_x1x2_1()


if __name__ == "__main__":
#    L_2()
#    L_3()
#    L_4()
    sphere = Sphere(10,10,10)
    points=sphere.points_in_poligon(1000) 
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d') #, aspect='equal'
    ii=0
    for point in points:
        lab = LabColor(point[0],point[1],point[2])
        xyz = convert_color(lab, sRGBColor)
        c0=xyz.get_rgb_hex()
        c1=c0[:7]
        ax.scatter(point[0], point[1], point[2], c=c1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()   
