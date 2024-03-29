
# Assignment1: Efficient Interpolation using Cubic Bezier and Tomas Algorithm.
## Preview
This repository contains a solution for an efficient interpolation task. The objective was to develop a function, Assignment1.interpolate(), that performs interpolation of a given function f within a specified range using a specific number of points. The goal was to achieve accurate interpolation while optimizing for efficiency.

## Solution
The implemented solution utilizes the <a href="https://en.wikipedia.org/wiki/Bézier_curve">
        Cubic Bezier Method
      </a> and <a href="https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm">
        Tomas Algorithm
      </a>. These techniques allow for effective interpolation, enabling accurate estimates of values between data points. The focus was on finding a balance between accuracy and computational efficiency.

<table>
  <tr>
    <td>
      <h2>Cubic Bezier</h2>
      <img src="https://github.com/ohayonyo/numerical_analysis_project/assets/62948137/7498090b-6a3a-49f5-8fc6-c28bbc059be3" alt="Cubic Bezier          Image" />
    </td>
    <td>
      <h2>Tomas Algorithm</h2>
      <img src="https://github.com/ohayonyo/numerical_analysis_project/assets/62948137/20fc597f-a1ff-4aef-a82a-f7ea233d7a5e" alt="Tomas Algorithm Image" width="298" />
    </td>
  </tr>
</table>

# Assignment2: Efficient Intersection points using Regula falsi. 
## Preview
In this repository, I present a solution to the Assignment2 task, which involved finding intersection points between two functions. The assignment required the implementation of the Assignment2.intersections() function, following the provided pydoc instructions.
## Solution
To solve this task, I used the <a href="https://en.wikipedia.org/wiki/Regula_falsi">
    Regula Falsi
 </a>. This numerical technique iteratively refines estimates of the intersection points by narrowing down the intervals where the functions' values change signs. By employing Regula Falsi, the solution is able to efficiently approximate intersection points while satisfying the provided conditions.

 

![regula](https://github.com/ohayonyo/numerical_analysis_project/assets/62948137/5841e5c4-1dc2-4e45-9ce6-7e45ed07be83)


## written by Yoad Ohayon
