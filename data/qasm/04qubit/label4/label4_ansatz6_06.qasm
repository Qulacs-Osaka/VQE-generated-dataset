OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.5163799375779496) q[0];
ry(2.5326651129608804) q[1];
cx q[0],q[1];
ry(-1.12588282466972) q[0];
ry(2.102740168116329) q[1];
cx q[0],q[1];
ry(-1.5036021426528436) q[1];
ry(0.04191177436137661) q[2];
cx q[1],q[2];
ry(3.1244728606477428) q[1];
ry(2.8961124529492484) q[2];
cx q[1],q[2];
ry(1.960151127196271) q[2];
ry(1.71663856601614) q[3];
cx q[2],q[3];
ry(-2.3293219746293397) q[2];
ry(-0.8210773855879369) q[3];
cx q[2],q[3];
ry(2.7274825873828035) q[0];
ry(-0.16800566431590624) q[1];
cx q[0],q[1];
ry(-2.326506730730128) q[0];
ry(0.17022901006682378) q[1];
cx q[0],q[1];
ry(2.410640068941438) q[1];
ry(-1.71063264533199) q[2];
cx q[1],q[2];
ry(0.5667518971229064) q[1];
ry(-2.3116981794856106) q[2];
cx q[1],q[2];
ry(-1.187000146936834) q[2];
ry(1.5354436126893125) q[3];
cx q[2],q[3];
ry(0.39810391469100015) q[2];
ry(1.6776135437546584) q[3];
cx q[2],q[3];
ry(2.0462567500459974) q[0];
ry(2.7486114643479493) q[1];
cx q[0],q[1];
ry(1.067390669580874) q[0];
ry(-2.1164574862622016) q[1];
cx q[0],q[1];
ry(2.174142284415249) q[1];
ry(2.452829814827846) q[2];
cx q[1],q[2];
ry(0.38495799671597375) q[1];
ry(-1.1620875978082417) q[2];
cx q[1],q[2];
ry(-1.6511607433151072) q[2];
ry(-0.7002383808382096) q[3];
cx q[2],q[3];
ry(-1.3602071224866892) q[2];
ry(0.30657667043609954) q[3];
cx q[2],q[3];
ry(2.882940884112472) q[0];
ry(2.0430412719637463) q[1];
cx q[0],q[1];
ry(0.4081126908907784) q[0];
ry(-0.2388673254865502) q[1];
cx q[0],q[1];
ry(0.1458262373384418) q[1];
ry(-0.6941939056147106) q[2];
cx q[1],q[2];
ry(-2.0021625789692177) q[1];
ry(0.5928027769400958) q[2];
cx q[1],q[2];
ry(1.946233353652763) q[2];
ry(-1.5045111478597137) q[3];
cx q[2],q[3];
ry(0.5663906341051291) q[2];
ry(-1.0162540883972433) q[3];
cx q[2],q[3];
ry(-2.8731678840655603) q[0];
ry(-0.3522309463647373) q[1];
cx q[0],q[1];
ry(-0.6149711567705662) q[0];
ry(2.18564027332528) q[1];
cx q[0],q[1];
ry(-1.4927394455148468) q[1];
ry(-1.6494137096827775) q[2];
cx q[1],q[2];
ry(2.7672965984189464) q[1];
ry(2.3239910442309824) q[2];
cx q[1],q[2];
ry(1.884330970970054) q[2];
ry(1.7564367409598742) q[3];
cx q[2],q[3];
ry(0.20574616482312802) q[2];
ry(-1.3614409611266334) q[3];
cx q[2],q[3];
ry(1.7101875197083156) q[0];
ry(0.39935329214391224) q[1];
cx q[0],q[1];
ry(-0.15741655178606795) q[0];
ry(-2.0051300459146413) q[1];
cx q[0],q[1];
ry(0.06572321350352135) q[1];
ry(0.7369111989596877) q[2];
cx q[1],q[2];
ry(0.18823225177738867) q[1];
ry(-2.063527351558762) q[2];
cx q[1],q[2];
ry(-1.51581808288379) q[2];
ry(3.0295570612586284) q[3];
cx q[2],q[3];
ry(-1.3763502044917475) q[2];
ry(3.0788905837311646) q[3];
cx q[2],q[3];
ry(1.1099436402103686) q[0];
ry(-1.3666186616754485) q[1];
cx q[0],q[1];
ry(-3.022981899685722) q[0];
ry(-1.079604000551943) q[1];
cx q[0],q[1];
ry(-0.24579480853074337) q[1];
ry(-2.5027302172043706) q[2];
cx q[1],q[2];
ry(-1.1774241573838442) q[1];
ry(-2.7627905030398767) q[2];
cx q[1],q[2];
ry(-0.46569697704973806) q[2];
ry(-2.7816897491050168) q[3];
cx q[2],q[3];
ry(0.4622834456943385) q[2];
ry(-1.5276386520139948) q[3];
cx q[2],q[3];
ry(1.6519029674433048) q[0];
ry(-0.8280346949187596) q[1];
cx q[0],q[1];
ry(-1.2486255150178085) q[0];
ry(-1.8017949071317254) q[1];
cx q[0],q[1];
ry(1.8767360106054527) q[1];
ry(0.3519974834615462) q[2];
cx q[1],q[2];
ry(-0.8204617000274991) q[1];
ry(-0.3179541438553047) q[2];
cx q[1],q[2];
ry(-0.9462062071957086) q[2];
ry(-2.248289500475642) q[3];
cx q[2],q[3];
ry(-3.0721344450847394) q[2];
ry(-2.0535943055632178) q[3];
cx q[2],q[3];
ry(-1.8542350175302083) q[0];
ry(-0.9618969241493689) q[1];
cx q[0],q[1];
ry(-2.280495963878926) q[0];
ry(2.2593979375648257) q[1];
cx q[0],q[1];
ry(-1.7996535531291669) q[1];
ry(-0.40679601744226346) q[2];
cx q[1],q[2];
ry(0.2299353676886264) q[1];
ry(-1.4218001361222958) q[2];
cx q[1],q[2];
ry(1.2259486001608595) q[2];
ry(2.8613091201855516) q[3];
cx q[2],q[3];
ry(-0.41952791435962045) q[2];
ry(2.930471018415733) q[3];
cx q[2],q[3];
ry(-2.826154329439744) q[0];
ry(2.6783611910522813) q[1];
ry(-1.562163097596905) q[2];
ry(0.0747808401488263) q[3];