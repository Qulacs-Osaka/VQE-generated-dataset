OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5759627086973593) q[0];
rz(-1.9923660838536763) q[0];
ry(-1.568122456589522) q[1];
rz(-0.0038189867350473246) q[1];
ry(0.003300262230058415) q[2];
rz(-1.3547911191107112) q[2];
ry(-2.950377536189152) q[3];
rz(0.07685207026118893) q[3];
ry(-1.5667017562008514) q[4];
rz(2.78787902150214) q[4];
ry(1.570185781235967) q[5];
rz(2.585091185360798) q[5];
ry(-6.167709086687626e-05) q[6];
rz(-1.0339992035270953) q[6];
ry(0.01115083543241796) q[7];
rz(2.0485940040127093) q[7];
ry(-0.23279609866166942) q[8];
rz(-0.3725755714536226) q[8];
ry(0.00017231317578969) q[9];
rz(-2.8490356844493805) q[9];
ry(-0.00018910081133149959) q[10];
rz(1.80451804355691) q[10];
ry(3.1415676542587083) q[11];
rz(2.4877013637453373) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.6941415112081843) q[0];
rz(-1.6802352399274056) q[0];
ry(1.792113800426719) q[1];
rz(1.6375441003704314) q[1];
ry(-1.5709286370707103) q[2];
rz(1.5732047954815316) q[2];
ry(0.03230590945152387) q[3];
rz(3.000896715695198) q[3];
ry(1.5081441037525738) q[4];
rz(-1.5882766595838804) q[4];
ry(-0.008087112679910744) q[5];
rz(0.46189872669376353) q[5];
ry(-3.1415410411328764) q[6];
rz(-1.3316866747920235) q[6];
ry(0.0027360118794517163) q[7];
rz(-0.8782840670001495) q[7];
ry(3.1246122638117226) q[8];
rz(-1.9436251069557036) q[8];
ry(3.1402371009599106) q[9];
rz(2.607988133814446) q[9];
ry(-3.572562934130746e-05) q[10];
rz(1.3013564176972554) q[10];
ry(0.0001547181752448168) q[11];
rz(0.12949365625562945) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.13733389967913) q[0];
rz(-2.0834977352750217) q[0];
ry(0.0024273276994214036) q[1];
rz(-0.0721930349814004) q[1];
ry(1.5638650516244357) q[2];
rz(-3.1407488163224464) q[2];
ry(3.1415681561121365) q[3];
rz(2.650034108438086) q[3];
ry(1.5697783023235397) q[4];
rz(2.4718348230769482) q[4];
ry(3.141492702009693) q[5];
rz(-0.15184086579478284) q[5];
ry(3.141477908274663) q[6];
rz(-0.0020324882865618557) q[6];
ry(-3.1406737093022494) q[7];
rz(1.3249456431251219) q[7];
ry(1.570265387893725) q[8];
rz(-0.8499747344817532) q[8];
ry(-1.5678387455919252) q[9];
rz(0.9389910897247642) q[9];
ry(1.5715730664024923) q[10];
rz(-0.0010030443954182045) q[10];
ry(-1.9357384019340569) q[11];
rz(-3.1218142260159967) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.140653623461367) q[0];
rz(3.048480514786064) q[0];
ry(-1.591059194156352) q[1];
rz(1.3516836352853643) q[1];
ry(1.5705766030721051) q[2];
rz(-1.5361004079348566) q[2];
ry(3.1397951999980016) q[3];
rz(1.1489218749224142) q[3];
ry(3.140720350616145) q[4];
rz(-0.6697063332340134) q[4];
ry(0.2222631034987632) q[5];
rz(-1.457860909357605) q[5];
ry(-3.141160372986887) q[6];
rz(2.7214179754128343) q[6];
ry(-1.5705092909413194) q[7];
rz(-3.1094184610480022) q[7];
ry(3.1403627720291993) q[8];
rz(1.3319369277055975) q[8];
ry(0.0007517140862948998) q[9];
rz(-1.3628390243065163) q[9];
ry(1.5719997074385752) q[10];
rz(0.13807883109458774) q[10];
ry(0.002047458914185007) q[11];
rz(-1.9265675166122547) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.3077471837412302) q[0];
rz(1.9859428366729261) q[0];
ry(-1.5701319175218558) q[1];
rz(-0.12506107286167722) q[1];
ry(-0.10948484299490621) q[2];
rz(2.848042033503785) q[2];
ry(3.1285022671598104) q[3];
rz(1.587973192813296) q[3];
ry(-1.5912924038905274) q[4];
rz(-1.570970955641843) q[4];
ry(-3.125570768709828) q[5];
rz(2.7992942766403814) q[5];
ry(-3.1412614165898973) q[6];
rz(-2.9857627462185716) q[6];
ry(-1.5728072636557842) q[7];
rz(-1.4375763434802713) q[7];
ry(0.003102939025729334) q[8];
rz(-1.298086215559304) q[8];
ry(3.138929543695769) q[9];
rz(2.1700351773576205) q[9];
ry(3.1413660610641205) q[10];
rz(0.13585231549859958) q[10];
ry(-3.139126619837786) q[11];
rz(0.7295967157707518) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.008619246794797523) q[0];
rz(1.2793419584423544) q[0];
ry(-3.1411108696639256) q[1];
rz(-0.10018514518318257) q[1];
ry(0.0011173408162523657) q[2];
rz(-2.8761229737758414) q[2];
ry(1.5745615558158659) q[3];
rz(-1.4287775337207789) q[3];
ry(-1.5868717398202383) q[4];
rz(1.5707738391692088) q[4];
ry(-0.002582200919865052) q[5];
rz(-2.7385443366904787) q[5];
ry(3.1410790860315454) q[6];
rz(-0.9118989687615747) q[6];
ry(0.009998281506544332) q[7];
rz(-2.500572278391021) q[7];
ry(0.00017593484654305434) q[8];
rz(-2.9371002211335617) q[8];
ry(0.7689610533353405) q[9];
rz(-0.803940873227889) q[9];
ry(0.6870559605186148) q[10];
rz(-1.5692865231633801) q[10];
ry(-3.1391793900615896) q[11];
rz(0.2449520583128351) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.6330177808027813) q[0];
rz(-1.8481508313389212) q[0];
ry(-0.00176195382776978) q[1];
rz(-1.53327656985335) q[1];
ry(-0.017410343099138362) q[2];
rz(-1.5774561581023203) q[2];
ry(-3.1399295484125718) q[3];
rz(-2.5303726512334896) q[3];
ry(-1.908387171319685) q[4];
rz(1.5710459310087574) q[4];
ry(-2.7730359655662333) q[5];
rz(0.7853415179482793) q[5];
ry(-3.1412277743888617) q[6];
rz(-0.7897391292027064) q[6];
ry(-3.1218547631870677) q[7];
rz(-2.5576302582278654) q[7];
ry(3.139984535096929) q[8];
rz(1.8106445296889255) q[8];
ry(3.140870635524027) q[9];
rz(-0.9945537382484261) q[9];
ry(1.5714065337226168) q[10];
rz(-1.570175056003852) q[10];
ry(3.1406018781818617) q[11];
rz(0.7491182307247833) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.140454205576677) q[0];
rz(1.15285567034166) q[0];
ry(0.0007628433636215348) q[1];
rz(3.0886420859935844) q[1];
ry(-1.5678398877252142) q[2];
rz(-1.5708108487529584) q[2];
ry(3.1413727463764856) q[3];
rz(-2.7073819665140557) q[3];
ry(1.5704291123588583) q[4];
rz(1.5616054811272961) q[4];
ry(0.0014908058816204654) q[5];
rz(2.3627340810101423) q[5];
ry(3.1415582497322982) q[6];
rz(0.9113578727585754) q[6];
ry(-0.0005687986480813478) q[7];
rz(0.8718285125837798) q[7];
ry(-0.004594847555146941) q[8];
rz(-3.031117479455482) q[8];
ry(3.1379919715686286) q[9];
rz(0.22335804550992122) q[9];
ry(1.5370120576923412) q[10];
rz(-1.5723429622190457) q[10];
ry(-1.5551735694240576) q[11];
rz(0.00016638792509062408) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.31947106438566814) q[0];
rz(2.3339009323207978) q[0];
ry(-1.5720118130592116) q[1];
rz(1.3497920758973052) q[1];
ry(1.8769074708253672) q[2];
rz(-2.6099383649787264) q[2];
ry(-2.973022446385718) q[3];
rz(-1.6054486408243545) q[3];
ry(-0.37062973989764275) q[4];
rz(1.581611672950979) q[4];
ry(-2.5481017284889327) q[5];
rz(1.5496008026464876) q[5];
ry(-3.141531124221878) q[6];
rz(-2.255462296212066) q[6];
ry(-0.002065396371893513) q[7];
rz(-1.3651393937212113) q[7];
ry(2.9927301428830977) q[8];
rz(0.0030100584697860693) q[8];
ry(-1.569264719527359) q[9];
rz(-1.5394105572174592) q[9];
ry(1.4062648790047971) q[10];
rz(-1.577539509510949) q[10];
ry(1.569665464261078) q[11];
rz(1.5965098218063813) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-8.590799829644169e-05) q[0];
rz(1.0310064531254195) q[0];
ry(3.141322383722277) q[1];
rz(-0.21922680070001646) q[1];
ry(-3.141036759378934) q[2];
rz(0.20977642615343076) q[2];
ry(1.5710786024694854) q[3];
rz(-3.140746031784926) q[3];
ry(1.5720198453963068) q[4];
rz(2.9959479724517895) q[4];
ry(3.1413922185004783) q[5];
rz(-0.0249264581654689) q[5];
ry(3.141018159572961) q[6];
rz(-2.46312428731811) q[6];
ry(0.0009628904164928898) q[7];
rz(-0.7498823269697453) q[7];
ry(-3.1413545884927716) q[8];
rz(2.3061702252833753) q[8];
ry(-1.5709134139824394) q[9];
rz(3.1413444961712447) q[9];
ry(-0.0004699175604319663) q[10];
rz(-3.1344023067482327) q[10];
ry(-1.5704819364492781) q[11];
rz(0.00011230943027802673) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415430297748625) q[0];
rz(-2.845991559376113) q[0];
ry(1.5711053128621773) q[1];
rz(-0.01456745403078852) q[1];
ry(-3.140820377215698) q[2];
rz(-0.597382308052148) q[2];
ry(1.5707673605969734) q[3];
rz(-2.5371870952923254) q[3];
ry(0.1585509243645049) q[4];
rz(-0.6536781357220185) q[4];
ry(-1.5707766231283014) q[5];
rz(2.1397787982039675) q[5];
ry(1.5707159986939585) q[6];
rz(3.141524829067538) q[6];
ry(-0.001434289745333655) q[7];
rz(0.8811315878295671) q[7];
ry(-1.5701449949377335) q[8];
rz(1.5697278847961273) q[8];
ry(1.5728052055331725) q[9];
rz(1.5711505327812025) q[9];
ry(-1.481957052359211) q[10];
rz(-3.14053259293714) q[10];
ry(1.5730606899190205) q[11];
rz(2.7052950142602463) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.141571933602459) q[0];
rz(1.9955967223968059) q[0];
ry(3.1413961458451203) q[1];
rz(1.5566670542340186) q[1];
ry(3.1414541611649045) q[2];
rz(-1.8460032084516462) q[2];
ry(-4.245769773447705e-05) q[3];
rz(0.5591903087891142) q[3];
ry(-3.141532369259984) q[4];
rz(-0.8419398306945061) q[4];
ry(-3.1413466982641145) q[5];
rz(1.434314776831024) q[5];
ry(1.5712413175269466) q[6];
rz(0.6175493246858698) q[6];
ry(-0.00046323615648050044) q[7];
rz(2.126917542540123) q[7];
ry(-1.5710827842344661) q[8];
rz(-1.5704826389659123) q[8];
ry(1.570593561102047) q[9];
rz(-3.141542904039091) q[9];
ry(-1.5709600205198493) q[10];
rz(-1.5703438878598526) q[10];
ry(-0.00014788078840588526) q[11];
rz(2.004481713508813) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5705807719642504) q[0];
rz(2.6834131868841826) q[0];
ry(1.5705435823949871) q[1];
rz(0.2271081365376158) q[1];
ry(-1.5712273927741105) q[2];
rz(1.1130640089730226) q[2];
ry(3.1380941612284343) q[3];
rz(2.9624531524597244) q[3];
ry(0.021127907558412318) q[4];
rz(1.1550296099973898) q[4];
ry(3.140545461391475) q[5];
rz(1.0730137480840847) q[5];
ry(-3.1410788027630945) q[6];
rz(1.7324650891449647) q[6];
ry(1.5682030240087395) q[7];
rz(0.22836309823917364) q[7];
ry(-1.5714344755537706) q[8];
rz(-0.45775027682004776) q[8];
ry(1.5712992730387114) q[9];
rz(-2.9122213135776893) q[9];
ry(1.5707592331057358) q[10];
rz(2.684701962259465) q[10];
ry(1.567874000126829) q[11];
rz(0.22761936101123312) q[11];