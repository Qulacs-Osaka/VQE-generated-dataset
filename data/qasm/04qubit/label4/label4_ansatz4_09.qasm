OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.8602252900682206) q[0];
rz(-3.0197078170262532) q[0];
ry(1.306402970800039) q[1];
rz(2.0390471367332443) q[1];
ry(-3.0807920521737233) q[2];
rz(-2.4959326081984172) q[2];
ry(3.092905815405084) q[3];
rz(2.502972379876302) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.546703404226297) q[0];
rz(1.0266563086214164) q[0];
ry(0.5106583456488814) q[1];
rz(-0.2509387832583476) q[1];
ry(-1.8765974705587452) q[2];
rz(-2.955512122857832) q[2];
ry(2.393542431751938) q[3];
rz(-0.4241102603472529) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.6050210614335054) q[0];
rz(1.9922589088322882) q[0];
ry(-2.6748113388691017) q[1];
rz(-2.276915802576641) q[1];
ry(0.49076894728576326) q[2];
rz(2.7057849921636796) q[2];
ry(0.7501431497210747) q[3];
rz(0.5709291919147959) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.628931665494981) q[0];
rz(1.2655356443087034) q[0];
ry(-0.12361572182504776) q[1];
rz(0.9549442759295735) q[1];
ry(0.350350670712837) q[2];
rz(-3.0173773394218086) q[2];
ry(-1.8735221318494588) q[3];
rz(0.045209736587090245) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.255973529668422) q[0];
rz(-1.9909179680504907) q[0];
ry(-2.9542540375141844) q[1];
rz(-1.6472140553517045) q[1];
ry(-2.8109461974574637) q[2];
rz(0.33720526786700344) q[2];
ry(-0.94743520057226) q[3];
rz(3.100263263557745) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7567093388473274) q[0];
rz(0.044264734443681064) q[0];
ry(-0.09037283196943857) q[1];
rz(-1.6359066465510652) q[1];
ry(-2.4352370926860076) q[2];
rz(-1.9659571284232702) q[2];
ry(2.961794115961534) q[3];
rz(0.9479434682102889) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.127376582730368) q[0];
rz(0.16972752659886936) q[0];
ry(1.1541303752430405) q[1];
rz(-1.7204343862775828) q[1];
ry(-2.1797302117887947) q[2];
rz(0.3409262206578925) q[2];
ry(-0.23590613635937085) q[3];
rz(1.296857918706316) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.1964703033675728) q[0];
rz(-2.8650967579227387) q[0];
ry(2.097154726799852) q[1];
rz(1.023534657946188) q[1];
ry(-1.1215893853745493) q[2];
rz(2.2284367512740553) q[2];
ry(0.5242031348512189) q[3];
rz(2.3928474934021735) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.667916982019406) q[0];
rz(-1.9261823664236601) q[0];
ry(-1.949143528221402) q[1];
rz(-2.8274414072786143) q[1];
ry(1.9851573497374178) q[2];
rz(-2.9490099704044304) q[2];
ry(-3.035403410923343) q[3];
rz(1.579536688680533) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.2141352562966626) q[0];
rz(0.0342561173826077) q[0];
ry(2.9036086388301325) q[1];
rz(-0.37890550594815137) q[1];
ry(2.9059318406315358) q[2];
rz(0.4260848130190444) q[2];
ry(0.07551311620215749) q[3];
rz(-1.1325372875853303) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.5676555804731587) q[0];
rz(2.3408730154151507) q[0];
ry(-1.9386135359241572) q[1];
rz(-2.5541713753187185) q[1];
ry(0.4563319992983999) q[2];
rz(0.32462063509792044) q[2];
ry(1.8611247751288866) q[3];
rz(-0.2008014559719945) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.2593696158547565) q[0];
rz(1.8729500368109657) q[0];
ry(0.3818811720900923) q[1];
rz(2.3702565271062848) q[1];
ry(-2.6080712809823465) q[2];
rz(-0.3583196712567991) q[2];
ry(2.366302246147669) q[3];
rz(-1.4946736772389526) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.9247390393182693) q[0];
rz(-2.951437890408302) q[0];
ry(0.3736808520614421) q[1];
rz(0.04152937218185263) q[1];
ry(0.7419855193834004) q[2];
rz(-2.1972713955329506) q[2];
ry(1.8842227652434531) q[3];
rz(2.4993527952788566) q[3];