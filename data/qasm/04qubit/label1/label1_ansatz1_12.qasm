OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.1751012488211974) q[0];
rz(-1.8216315150638946) q[0];
ry(1.7372639379448565) q[1];
rz(-1.2440895644850247) q[1];
ry(-0.1464822396301178) q[2];
rz(1.185571249884811) q[2];
ry(2.393521042107113) q[3];
rz(-0.5465567530096482) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.021948684225685) q[0];
rz(0.4736485981420202) q[0];
ry(2.387269929604987) q[1];
rz(-2.850036664025085) q[1];
ry(0.05134858876805211) q[2];
rz(0.37198510715422517) q[2];
ry(-0.5741394905378243) q[3];
rz(1.5278821305545716) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.1842903732148655) q[0];
rz(1.0193582899413054) q[0];
ry(0.08345190365499784) q[1];
rz(-1.1813917928084168) q[1];
ry(2.56230860076222) q[2];
rz(-2.5344600149841865) q[2];
ry(1.7165723576336394) q[3];
rz(-1.0304478710495781) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.054916585413613) q[0];
rz(-1.22595533929026) q[0];
ry(2.8318904852222593) q[1];
rz(0.20999726626932524) q[1];
ry(2.5125134289637057) q[2];
rz(-2.3237389957028034) q[2];
ry(-2.8690912480553843) q[3];
rz(1.933198538174145) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.9936837201919992) q[0];
rz(-2.8450247504103796) q[0];
ry(0.783109686467987) q[1];
rz(1.114576970368769) q[1];
ry(-0.2423639588634492) q[2];
rz(0.9968593804544011) q[2];
ry(-1.7796793714220545) q[3];
rz(-1.3681473789951522) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.678347202197339) q[0];
rz(2.973171981318717) q[0];
ry(1.9545950355586692) q[1];
rz(-2.732078693745049) q[1];
ry(-0.45810628425363037) q[2];
rz(-0.04750613098096205) q[2];
ry(1.1426290926504834) q[3];
rz(-2.8666646081435583) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8052637676619274) q[0];
rz(-2.5568223131290777) q[0];
ry(-1.4689938194389223) q[1];
rz(-1.8311221880774164) q[1];
ry(0.08043193115536784) q[2];
rz(1.8491001800296774) q[2];
ry(-0.7575482887896249) q[3];
rz(1.49906998131834) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.652264675944017) q[0];
rz(-1.7023099306208636) q[0];
ry(-0.3780558540057033) q[1];
rz(0.6300523032871791) q[1];
ry(-2.9344281177507727) q[2];
rz(0.5361086759067478) q[2];
ry(-1.0443226301790887) q[3];
rz(-2.9653523506313597) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.9418872499411304) q[0];
rz(1.5539651399499834) q[0];
ry(-2.8736782603117623) q[1];
rz(2.3346784708429604) q[1];
ry(-1.3803422582952969) q[2];
rz(2.166927382148594) q[2];
ry(1.4573089784878983) q[3];
rz(-2.353460005106529) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.1299359736878063) q[0];
rz(2.070810624670644) q[0];
ry(-0.6280719817295841) q[1];
rz(-1.4108304329421764) q[1];
ry(1.9016837456445863) q[2];
rz(-1.6114977680809028) q[2];
ry(1.759794693193105) q[3];
rz(-1.1264516589428375) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.696167567310843) q[0];
rz(-1.1256770966345346) q[0];
ry(-2.1938748015693523) q[1];
rz(-0.6923900299076848) q[1];
ry(1.5358260459317075) q[2];
rz(-0.8515438941489729) q[2];
ry(-1.5218435419720684) q[3];
rz(0.9599691816617673) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.1272370261284177) q[0];
rz(-1.5604642281608676) q[0];
ry(2.524098428392925) q[1];
rz(1.6132372995937443) q[1];
ry(2.337511026388803) q[2];
rz(2.198665613488916) q[2];
ry(2.9487338677364328) q[3];
rz(3.0037487327034347) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.712567278290668) q[0];
rz(-2.3942580829416253) q[0];
ry(-2.4642708026460647) q[1];
rz(-0.5966694776096725) q[1];
ry(1.6679814532475037) q[2];
rz(-0.010119626776632994) q[2];
ry(-0.5510202636932781) q[3];
rz(1.1481738394723053) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.8614356560380969) q[0];
rz(2.7762119620557506) q[0];
ry(1.646894269068721) q[1];
rz(0.8408489465704897) q[1];
ry(2.4178443865579) q[2];
rz(1.4555653786997298) q[2];
ry(-0.2027712662627436) q[3];
rz(0.9449953211649871) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7951991408333564) q[0];
rz(2.6699307119684557) q[0];
ry(1.8016196928818644) q[1];
rz(-3.020692495763948) q[1];
ry(2.081383891186902) q[2];
rz(2.520380367492146) q[2];
ry(1.0822434790025737) q[3];
rz(-1.4775723266418348) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6317756939891059) q[0];
rz(2.7608240911745927) q[0];
ry(1.5939512783235614) q[1];
rz(2.0344243283696812) q[1];
ry(2.1303851857489384) q[2];
rz(3.070638778966484) q[2];
ry(-1.6430103033377004) q[3];
rz(-2.741119451126501) q[3];