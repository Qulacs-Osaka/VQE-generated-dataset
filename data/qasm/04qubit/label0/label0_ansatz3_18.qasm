OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.026761444391213) q[0];
rz(1.163609171688744) q[0];
ry(2.7413092049103933) q[1];
rz(0.6399772490161694) q[1];
ry(0.6589111899579905) q[2];
rz(-0.6542967127048068) q[2];
ry(1.529887095153291) q[3];
rz(1.0205848272861275) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.4355356947905564) q[0];
rz(0.3194345191513488) q[0];
ry(-1.9280175027730304) q[1];
rz(1.5624374552052671) q[1];
ry(-0.47839361813824244) q[2];
rz(-0.2765057315944368) q[2];
ry(0.09776051082718987) q[3];
rz(2.9091346283805835) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0379688581907005) q[0];
rz(-1.5057426000567045) q[0];
ry(-0.37026533859364624) q[1];
rz(0.9301037938632106) q[1];
ry(-0.25222557772544135) q[2];
rz(-0.9192478010002768) q[2];
ry(-1.43844772944782) q[3];
rz(2.6225766376187236) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.1695098317902515) q[0];
rz(1.1350416226696318) q[0];
ry(-3.0593486211701753) q[1];
rz(-2.704879567717879) q[1];
ry(-2.468482734689508) q[2];
rz(-1.1281817014280962) q[2];
ry(-1.2665468234402528) q[3];
rz(-1.651353719355587) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.0716717133734965) q[0];
rz(0.09127520605027327) q[0];
ry(-2.7301383421124714) q[1];
rz(-2.219260520888843) q[1];
ry(2.948763376804091) q[2];
rz(-2.1132030875869536) q[2];
ry(-2.7667188518801837) q[3];
rz(2.5396632867965625) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.6190868986056683) q[0];
rz(-1.466488317650759) q[0];
ry(2.0145872823563367) q[1];
rz(-2.501676399145083) q[1];
ry(-2.1012689795533754) q[2];
rz(-1.4744203260290183) q[2];
ry(3.080031198973694) q[3];
rz(-1.6904904857705896) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.7916015602413857) q[0];
rz(2.0811323808665927) q[0];
ry(2.9225288785925487) q[1];
rz(-2.7203095593790407) q[1];
ry(-0.9379828330198283) q[2];
rz(-1.0303828463483322) q[2];
ry(1.5653405366625561) q[3];
rz(0.6610666678595427) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.0430098170899043) q[0];
rz(2.00784032379933) q[0];
ry(-2.537475930649366) q[1];
rz(-2.9624226323988516) q[1];
ry(-0.8391515351400529) q[2];
rz(-2.4073323236900306) q[2];
ry(2.6270812731208952) q[3];
rz(2.801778756113041) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.211299855902276) q[0];
rz(0.8997611152217209) q[0];
ry(2.9750078490795784) q[1];
rz(1.3525493813616247) q[1];
ry(1.794641623469697) q[2];
rz(-0.08071202458046978) q[2];
ry(0.552475199411453) q[3];
rz(-0.5415528262875632) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.446643401683932) q[0];
rz(-1.2424495693235211) q[0];
ry(1.928444000067615) q[1];
rz(2.6399603313622273) q[1];
ry(-1.587008689487618) q[2];
rz(0.05167330281929061) q[2];
ry(-3.08536243274908) q[3];
rz(0.5985120322031605) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.3091673697226085) q[0];
rz(0.9799789333003313) q[0];
ry(0.971488003308778) q[1];
rz(0.9931465650193383) q[1];
ry(0.427057199620998) q[2];
rz(2.6614118246978586) q[2];
ry(-2.2019900313203102) q[3];
rz(0.5115871735721251) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.220162928579568) q[0];
rz(-1.4419668819843718) q[0];
ry(1.1714682419861377) q[1];
rz(2.176242050702726) q[1];
ry(-2.932753544270087) q[2];
rz(0.2314769827151358) q[2];
ry(0.6293488729122858) q[3];
rz(-2.343564058617122) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.5307732230302733) q[0];
rz(-1.5875941395381838) q[0];
ry(0.6604489530364958) q[1];
rz(1.4935349996764957) q[1];
ry(-0.9853403810929819) q[2];
rz(2.185662727251271) q[2];
ry(-2.890873892810587) q[3];
rz(-2.0108568733745944) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.1269263901725788) q[0];
rz(-0.3389178435550262) q[0];
ry(-0.8100712852299319) q[1];
rz(-2.6738898309874646) q[1];
ry(-0.23431011877612296) q[2];
rz(3.108696287401987) q[2];
ry(2.2782775436391223) q[3];
rz(0.7265287102409301) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.6881290476295785) q[0];
rz(-2.6344056257477977) q[0];
ry(0.6688760891399914) q[1];
rz(0.38701849397325283) q[1];
ry(0.565036628188321) q[2];
rz(2.81796916568702) q[2];
ry(-2.684551480230446) q[3];
rz(-2.2172446604304197) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.6964691454266987) q[0];
rz(1.1948602619809972) q[0];
ry(-2.3287372024393846) q[1];
rz(-1.2203935937139105) q[1];
ry(1.327167405299086) q[2];
rz(2.0668481040604405) q[2];
ry(0.7049657255193997) q[3];
rz(-1.6190307115213738) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.58450000924393) q[0];
rz(1.5600954384676884) q[0];
ry(-0.7305750036745113) q[1];
rz(1.1083106685639805) q[1];
ry(0.5532127826093708) q[2];
rz(0.9107332294002887) q[2];
ry(-1.0925185631882859) q[3];
rz(-2.8277258705324635) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9485951045939212) q[0];
rz(1.821337017304936) q[0];
ry(-2.1152357855321173) q[1];
rz(-2.3857971325289746) q[1];
ry(-1.0730284256076619) q[2];
rz(-1.1961594081557951) q[2];
ry(-2.3599708465906595) q[3];
rz(2.0855494098917715) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.8238284964880889) q[0];
rz(2.5949243989902118) q[0];
ry(0.3778333755773806) q[1];
rz(-2.732174524238677) q[1];
ry(-0.49493315001257343) q[2];
rz(-1.7313933301345603) q[2];
ry(2.126422481391436) q[3];
rz(0.10951557748269725) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6332774283612261) q[0];
rz(-1.5342756312583576) q[0];
ry(-0.7558247061315592) q[1];
rz(2.7320289054549334) q[1];
ry(-1.8914353513885869) q[2];
rz(-1.4107982316749652) q[2];
ry(1.0889625315167413) q[3];
rz(-2.730940532297825) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.21751874611005342) q[0];
rz(-2.659268290563804) q[0];
ry(1.421613131105568) q[1];
rz(1.3480154789856296) q[1];
ry(-2.9924105590178476) q[2];
rz(1.876963540252686) q[2];
ry(2.596303705890615) q[3];
rz(0.8401660875694272) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.101282776812713) q[0];
rz(-1.7844515918811295) q[0];
ry(0.460095192229873) q[1];
rz(2.547318124346364) q[1];
ry(-2.9330688078555367) q[2];
rz(1.7334162655528988) q[2];
ry(-0.45785719995807306) q[3];
rz(-0.4425560889635429) q[3];