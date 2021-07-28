#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*h = 10.^-10;*)
(*x0 = {1.5, 1.5};*)
(*\[Delta] = 10.^-5;*)
(*Kmax = 10;*)


(* ::Input:: *)
(*Clear[f1]*)
(*f1[x_] :=*)
(*    x[[1]]^2 - x[[2]]^2 - 1*)


(* ::Input:: *)
(*Clear[f2]*)
(*f2[x_] :=*)
(*    x[[1]] x[[2]]^3 - x[[2]] - 1*)


(* ::Input:: *)
(*Clear[F]*)
(*F[x_] := {*)
(*    f1[x], f2[x]*)
(*}*)


(* ::Input:: *)
(*Clear[secantMethod]*)
(*secantMethod[F_Symbol, x0_List, Kmax_Integer, \[Delta]_Real, h_Real] :=*)
(*    Module[{x = {x0}, k = 1, dRes = \[Delta], H, G, fDInv, e, n},*)
(*        n = Length[F[x[[1]]]];*)
(*        H = h * IdentityMatrix[n];*)
(*        While[k <= Kmax \[And] \[Delta] <= dRes,*)
(*            G = Table[*)
(*                e = ConstantArray[0, n];*)
(*                e[[j]] = 1;*)
(*                F[x[[k]] + h * e][[i]] - F[x[[k]]][[i]],*)
(*                {i, n}, {j, n}*)
(*            ];*)
(*            fDInv = Inverse[G] . H;*)
(*            AppendTo[x, x[[k]] - fDInv . F[x[[k]]]];*)
(*            dRes = Norm[x[[k + 1]] - x[[k]]];*)
(*            k += 1*)
(*        ];*)
(*        x[[-1]]*)
(*    ]*)


(* ::Input:: *)
(*secantMethod[F, x0, Kmax, \[Delta], h]*)


(* ::Input:: *)
(*NSolve[{x1^2 - x2^2 - 1 == 0, x1 x2^3 - x2 - 1 == 0}, {x1, x2}, Reals]*)


(* ::Input:: *)
(*x = secantMethod[F, x0, Kmax, \[Delta], h]*)


(* ::Input:: *)
(*r = F[x]*)
