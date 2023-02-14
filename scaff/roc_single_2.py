"""
Generates images of single-point ROC curves
"""

import matplotlib.pyplot as plt

class SRoc:
    "draws ROC space"
    
    def draw_curve(self, fp, tp, c = "b"):
        ax = plt.axes()
        plt.plot([-0.001,1.001],[-0.001,1.001],color="orange") # diagonal reference
        plt.plot((fp,), (tp,), "s"+c)
#        plt.plot((0, fp, 1), (0, tp, 1))
        plt.plot((0, fp, 1), (0, tp, 1), "-"+c)
        ax.set_aspect('equal')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
#        plt.show()

    def draw_point(self, fp, tp):
        ax = plt.axes()
        plt.plot([-0.001,1.001],[-0.001,1.001],color="orange") # diagonal reference
        plt.plot((fp,), (tp,), "sb")
        plt.plot((fp, fp), (0,tp), ":b")
        plt.plot((0, fp), (tp,tp), ":b")
        ax.set_aspect('equal')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.show()

if __name__ == "__main__":

    r = SRoc()

#    r.draw_point(0.3, 0.7)

    r.draw_curve(0.15, 0.65)
    r.draw_curve(0.55, 0.9, "g")
    plt.show()
