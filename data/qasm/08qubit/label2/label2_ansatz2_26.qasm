OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.5233309357188238) q[0];
rz(2.599404067199798) q[0];
ry(3.094261031945743) q[1];
rz(0.24505256110304274) q[1];
ry(0.18856109888103578) q[2];
rz(-1.1399685598141849) q[2];
ry(-2.7402051654091073) q[3];
rz(0.7614421579199009) q[3];
ry(-0.46480891364932114) q[4];
rz(-0.6431398046744289) q[4];
ry(-2.0436156858839825) q[5];
rz(-1.3188007027323447) q[5];
ry(1.4403273470127997) q[6];
rz(0.5714143403564158) q[6];
ry(1.6091523137450519) q[7];
rz(-2.240039600288971) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.393019178911886) q[0];
rz(-3.002464526853143) q[0];
ry(-0.45475958952914264) q[1];
rz(-2.604269961958494) q[1];
ry(0.7588549107455604) q[2];
rz(1.8534014003555903) q[2];
ry(1.2170160406696162) q[3];
rz(-1.9186019227674684) q[3];
ry(-3.1141117601098083) q[4];
rz(2.4128037346394846) q[4];
ry(1.4728699491364936) q[5];
rz(-1.9481776634532642) q[5];
ry(-0.8391860500630557) q[6];
rz(0.5442960082242054) q[6];
ry(1.3865905128183718) q[7];
rz(0.2633365130189116) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.83207658604399) q[0];
rz(-2.2069132631637425) q[0];
ry(3.0716594105693478) q[1];
rz(-1.476520264808003) q[1];
ry(-1.7417660362113274) q[2];
rz(1.8517171745791592) q[2];
ry(0.6280661090213046) q[3];
rz(-0.29687714630553713) q[3];
ry(3.0009110798368828) q[4];
rz(2.123312777773946) q[4];
ry(-2.0350191518923815) q[5];
rz(-0.037480200380873636) q[5];
ry(-0.2738391009654033) q[6];
rz(-0.6642899098521786) q[6];
ry(-0.8029437806509835) q[7];
rz(-0.9506023617959867) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.0069351161927331) q[0];
rz(1.0980804817155692) q[0];
ry(0.9738418512110103) q[1];
rz(2.124395033242814) q[1];
ry(-0.240703712302464) q[2];
rz(2.636875400031315) q[2];
ry(1.4694469111978972) q[3];
rz(-2.0107592357021193) q[3];
ry(0.4664685183012984) q[4];
rz(-0.578902915244032) q[4];
ry(1.8991710688637147) q[5];
rz(2.8647440648236087) q[5];
ry(-1.6882440218145593) q[6];
rz(1.2527830439596102) q[6];
ry(-1.9930905214186536) q[7];
rz(1.8955003737002647) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.0531347059018765) q[0];
rz(-1.2078700799642617) q[0];
ry(-3.077323139474132) q[1];
rz(2.100943017401671) q[1];
ry(0.46498545346762493) q[2];
rz(-1.3166118859805005) q[2];
ry(1.4360913925891712) q[3];
rz(-2.6586428009137975) q[3];
ry(-0.6267336897891087) q[4];
rz(1.3810408454036587) q[4];
ry(-0.4448519096395396) q[5];
rz(-2.4546656060586693) q[5];
ry(1.3718002418468265) q[6];
rz(0.2768390461886821) q[6];
ry(0.7627089697211122) q[7];
rz(1.4714580148524892) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.19916884119118985) q[0];
rz(-0.6507044366466905) q[0];
ry(-1.5719165470755418) q[1];
rz(0.35024659226554167) q[1];
ry(-2.254173009273852) q[2];
rz(2.1295826959585735) q[2];
ry(-1.6811515358952709) q[3];
rz(2.3751589826341766) q[3];
ry(-1.1001625604372514) q[4];
rz(-3.006691152307435) q[4];
ry(-0.035779609988552785) q[5];
rz(2.9809029774213016) q[5];
ry(-0.9275721639942196) q[6];
rz(0.12612156872173944) q[6];
ry(-0.6163413537025546) q[7];
rz(-1.2981929420109868) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.3299282275382853) q[0];
rz(-0.890776490799314) q[0];
ry(2.2998721366105848) q[1];
rz(-2.0007484363092214) q[1];
ry(0.513506239662704) q[2];
rz(-0.45096035122733036) q[2];
ry(2.4469157649116906) q[3];
rz(-0.2967704316713577) q[3];
ry(2.6838342140385714) q[4];
rz(2.902649882497169) q[4];
ry(2.2705490374454036) q[5];
rz(0.8892628468488507) q[5];
ry(-0.8776699042395288) q[6];
rz(2.024672311041118) q[6];
ry(-2.4410978051889547) q[7];
rz(-2.6276306978325965) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.6065470864997944) q[0];
rz(-1.7182897507309232) q[0];
ry(1.3989308606365505) q[1];
rz(-0.5404004047730461) q[1];
ry(-1.429182943359879) q[2];
rz(3.0077507831105255) q[2];
ry(-0.7831724695040212) q[3];
rz(-3.0761511930177674) q[3];
ry(1.3607278316510056) q[4];
rz(-2.475149275039592) q[4];
ry(-1.1012010421496576) q[5];
rz(-0.07893304138905463) q[5];
ry(-3.103807123521904) q[6];
rz(-1.2345667223381962) q[6];
ry(-0.050518838219404394) q[7];
rz(1.315814181511718) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.6753675636816614) q[0];
rz(1.1138531244042724) q[0];
ry(-1.076229423237179) q[1];
rz(0.543694814020945) q[1];
ry(-1.1271282978079817) q[2];
rz(1.786110230478495) q[2];
ry(2.6085957531484616) q[3];
rz(0.5654791287007813) q[3];
ry(-0.5255057557867785) q[4];
rz(-2.9063255864781263) q[4];
ry(-1.680814027674428) q[5];
rz(0.3991320999102505) q[5];
ry(2.1785468789050846) q[6];
rz(-0.010425274607340993) q[6];
ry(0.7618781433044051) q[7];
rz(0.1188249674715278) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.3630337929037282) q[0];
rz(-0.8218975019995813) q[0];
ry(-3.049053232241573) q[1];
rz(2.756357476361773) q[1];
ry(1.9081158296772327) q[2];
rz(-0.9524534066656872) q[2];
ry(-2.600441266873785) q[3];
rz(0.9441438651486173) q[3];
ry(1.1154018959273573) q[4];
rz(-2.240109151399463) q[4];
ry(-0.9177932903746413) q[5];
rz(2.2769145871846406) q[5];
ry(0.7133271227107878) q[6];
rz(1.902098014900985) q[6];
ry(0.5086155315979887) q[7];
rz(0.12882154660303127) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.8411761316359492) q[0];
rz(-0.9659143351082935) q[0];
ry(1.0990652408762305) q[1];
rz(2.980641713889908) q[1];
ry(0.24438541434298022) q[2];
rz(-1.8481453342819627) q[2];
ry(-0.10640858867927198) q[3];
rz(-1.8986243384593562) q[3];
ry(-1.625520579092185) q[4];
rz(0.7210026395423021) q[4];
ry(0.5234029793148299) q[5];
rz(-0.5765614038370981) q[5];
ry(-0.5204848405241639) q[6];
rz(0.11404895392249106) q[6];
ry(-2.3213383870370303) q[7];
rz(0.8920535455378555) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.2686241078740661) q[0];
rz(1.8060923626383667) q[0];
ry(1.8663670848207365) q[1];
rz(0.12587061226954532) q[1];
ry(1.4030072035105112) q[2];
rz(0.18140205153479913) q[2];
ry(-1.7609554580739033) q[3];
rz(-0.8366729154500565) q[3];
ry(2.3565909362904267) q[4];
rz(-1.6475894165366238) q[4];
ry(-0.8953521806208053) q[5];
rz(-2.402137212849881) q[5];
ry(0.371567073604541) q[6];
rz(-0.10278322614020396) q[6];
ry(0.2723622489507796) q[7];
rz(-1.3846771186190472) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.5422242504399337) q[0];
rz(2.8297046611327175) q[0];
ry(0.08788738024123344) q[1];
rz(2.4144319648071146) q[1];
ry(0.9755406738383154) q[2];
rz(3.0111014512517422) q[2];
ry(-2.0115211404315323) q[3];
rz(-2.8131071118057034) q[3];
ry(1.5025513487796356) q[4];
rz(2.0517225757649324) q[4];
ry(-2.74495807423668) q[5];
rz(1.557563624373038) q[5];
ry(-2.267558274003293) q[6];
rz(0.7679090233706449) q[6];
ry(-1.0423986700184513) q[7];
rz(0.8124334282231694) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.8193093405910367) q[0];
rz(0.3295947584399363) q[0];
ry(1.3832094297596038) q[1];
rz(-2.708034771999287) q[1];
ry(1.3665151872160097) q[2];
rz(-2.865197299083247) q[2];
ry(-1.1462206994180713) q[3];
rz(2.862764489852332) q[3];
ry(-2.6451637898658125) q[4];
rz(2.2802486028542646) q[4];
ry(0.5916298824780701) q[5];
rz(-2.9844973714894) q[5];
ry(1.9135365571737513) q[6];
rz(-0.999236640128756) q[6];
ry(-1.1962174608240952) q[7];
rz(1.0481857129462766) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.7264625366575093) q[0];
rz(-2.368532859046967) q[0];
ry(-2.284790162791561) q[1];
rz(3.0327633681261736) q[1];
ry(1.522700516828376) q[2];
rz(-1.4915315576632064) q[2];
ry(-2.236120845151015) q[3];
rz(-0.24667204988846497) q[3];
ry(-0.13960653764743097) q[4];
rz(-1.5115396616992287) q[4];
ry(-2.3111559656133025) q[5];
rz(-0.17541134530497138) q[5];
ry(2.9411638133340183) q[6];
rz(1.581000948380607) q[6];
ry(-2.2013298772899703) q[7];
rz(-0.5990212077232543) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.7766181857719934) q[0];
rz(2.2531149361118588) q[0];
ry(-2.761048803490261) q[1];
rz(0.20917280954225248) q[1];
ry(2.762737566311496) q[2];
rz(3.1370701287529794) q[2];
ry(2.8918604297214685) q[3];
rz(-2.590114646398837) q[3];
ry(-1.1759990955129374) q[4];
rz(-3.0948261993625463) q[4];
ry(-3.0315810783242183) q[5];
rz(-0.671190710421291) q[5];
ry(-1.553488861986304) q[6];
rz(-1.5398497266950457) q[6];
ry(-2.7409920604265787) q[7];
rz(-1.7704010728164985) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.8892114429079157) q[0];
rz(2.804620748746356) q[0];
ry(0.3815640410363382) q[1];
rz(-1.68552466688898) q[1];
ry(1.804124529480843) q[2];
rz(-0.44194413265383986) q[2];
ry(1.6140595800580197) q[3];
rz(2.9775834044035783) q[3];
ry(-2.533063368624996) q[4];
rz(-1.060754822564891) q[4];
ry(1.8857311047466836) q[5];
rz(0.6635441078461182) q[5];
ry(-1.7053182794387967) q[6];
rz(-0.20005200246788488) q[6];
ry(-0.28337171589325205) q[7];
rz(1.4279369339105217) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.7803248607279691) q[0];
rz(-1.6013486774866115) q[0];
ry(-1.4830576260129362) q[1];
rz(0.7946766751543747) q[1];
ry(1.7146991958106188) q[2];
rz(-0.3651085554279625) q[2];
ry(-2.73716911638842) q[3];
rz(1.525428086871684) q[3];
ry(-1.2317695627800997) q[4];
rz(2.219277440368371) q[4];
ry(3.070540974692296) q[5];
rz(-2.5658941621822384) q[5];
ry(1.3524384654593984) q[6];
rz(0.6846536710185287) q[6];
ry(2.6084513126492874) q[7];
rz(-0.10215261693163935) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.0405703773451371) q[0];
rz(-2.6602437614321865) q[0];
ry(-0.5988210451818174) q[1];
rz(-2.263224122602671) q[1];
ry(2.009933689104579) q[2];
rz(-0.5011783222708537) q[2];
ry(0.8375121919454972) q[3];
rz(3.085407780477308) q[3];
ry(-2.5159804389115394) q[4];
rz(-1.8085881429712867) q[4];
ry(2.325760889598851) q[5];
rz(2.0107631936251362) q[5];
ry(0.8675590264788533) q[6];
rz(2.108838062253552) q[6];
ry(2.7566821346436923) q[7];
rz(2.3293242757721218) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.778000319365295) q[0];
rz(0.2635103311036442) q[0];
ry(2.560416816446981) q[1];
rz(1.0664500902915268) q[1];
ry(2.872648801895548) q[2];
rz(-0.6197246624965281) q[2];
ry(-2.006269202380918) q[3];
rz(-0.07969753868098284) q[3];
ry(-0.11479098811098834) q[4];
rz(-2.8565436191032836) q[4];
ry(-1.8599237716755994) q[5];
rz(1.4576122726821223) q[5];
ry(1.25163471642444) q[6];
rz(1.3064268188839465) q[6];
ry(-1.6213651369909807) q[7];
rz(-0.627628872473756) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.7428618900240718) q[0];
rz(-0.043902667982148806) q[0];
ry(-0.07649637051100555) q[1];
rz(-1.3835995806535355) q[1];
ry(0.4076084940647705) q[2];
rz(2.2035823258137004) q[2];
ry(-0.09244451976916894) q[3];
rz(-2.903009504841255) q[3];
ry(0.8013891667193108) q[4];
rz(2.288117530408067) q[4];
ry(-0.8802552826346597) q[5];
rz(1.9698724354606512) q[5];
ry(-0.9252389624881454) q[6];
rz(0.030361631097388653) q[6];
ry(-1.2038770311699807) q[7];
rz(1.5693691204201736) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.384167465837027) q[0];
rz(1.3329145526107604) q[0];
ry(1.687882498114874) q[1];
rz(-2.549353439087292) q[1];
ry(-2.794950399961103) q[2];
rz(2.7579482576594643) q[2];
ry(-1.1524990177443692) q[3];
rz(-0.7641642358188614) q[3];
ry(-0.8959180200483949) q[4];
rz(1.2850219766415432) q[4];
ry(-2.27882622034213) q[5];
rz(-2.8361744846493524) q[5];
ry(2.611950230278558) q[6];
rz(-1.2121495094856853) q[6];
ry(-2.8567273022969744) q[7];
rz(-1.1672124962810706) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.330262854225285) q[0];
rz(3.0935679886933376) q[0];
ry(-1.563325238165258) q[1];
rz(-1.893101816500361) q[1];
ry(-1.122017033421144) q[2];
rz(-1.2446907969995875) q[2];
ry(0.8094464585931069) q[3];
rz(0.5015220060819461) q[3];
ry(1.5565046757140577) q[4];
rz(-0.70819116253299) q[4];
ry(1.4660675459963304) q[5];
rz(1.129013153706687) q[5];
ry(1.9617815532081915) q[6];
rz(2.8046108366116) q[6];
ry(1.2577927319825566) q[7];
rz(0.2986429963364232) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.36396249977225814) q[0];
rz(-0.408762491075553) q[0];
ry(-2.2233279339127074) q[1];
rz(-2.138533578563638) q[1];
ry(0.730100723285216) q[2];
rz(2.0951911904535967) q[2];
ry(-1.0729020730764436) q[3];
rz(-2.612947642705992) q[3];
ry(-1.344958975673118) q[4];
rz(-0.09439244971537814) q[4];
ry(-1.6340511602595862) q[5];
rz(3.00466417273622) q[5];
ry(2.2626229814477403) q[6];
rz(0.7362184307281343) q[6];
ry(2.041820018267731) q[7];
rz(2.8512044573719035) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.8362956003150548) q[0];
rz(-0.40681569723092803) q[0];
ry(-1.3930938988573005) q[1];
rz(2.4556809956737102) q[1];
ry(0.3642274447389182) q[2];
rz(1.7616842417857719) q[2];
ry(1.2943773890747206) q[3];
rz(2.7900282339690503) q[3];
ry(2.01027857279224) q[4];
rz(-1.506008190777914) q[4];
ry(2.3925742502907754) q[5];
rz(0.11855532077310986) q[5];
ry(2.5393870912596497) q[6];
rz(0.11093189370472521) q[6];
ry(-0.20896471244096523) q[7];
rz(-0.4051187508673241) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.819957540219193) q[0];
rz(0.22458528275327438) q[0];
ry(0.8802884374400969) q[1];
rz(-1.487916548215696) q[1];
ry(1.9507048253506243) q[2];
rz(-2.0324270063319867) q[2];
ry(-0.48015689738119166) q[3];
rz(-3.1010124810049144) q[3];
ry(1.8543577280232677) q[4];
rz(-1.320663209116087) q[4];
ry(0.9076598835445191) q[5];
rz(0.4552416570698944) q[5];
ry(-0.36969891709958536) q[6];
rz(-2.3214650090258186) q[6];
ry(-2.8523986580350345) q[7];
rz(-1.6666261467399686) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.6743304010492208) q[0];
rz(0.2781741283351015) q[0];
ry(0.3179977452002456) q[1];
rz(1.0449413615845726) q[1];
ry(0.425046886492475) q[2];
rz(-0.9893551374397324) q[2];
ry(-0.7433260819179146) q[3];
rz(0.3713626984378654) q[3];
ry(-1.2023522748125455) q[4];
rz(1.1034926221841521) q[4];
ry(0.5212772928255994) q[5];
rz(1.6986145086686246) q[5];
ry(1.2962002973209543) q[6];
rz(-2.130503619036868) q[6];
ry(0.7589660379813995) q[7];
rz(-2.4833129501670888) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-0.45434783186569805) q[0];
rz(2.9134974665784497) q[0];
ry(0.7951509248667588) q[1];
rz(-1.3498644240342437) q[1];
ry(-1.117606065435498) q[2];
rz(-1.7603158333357998) q[2];
ry(0.26152656107543937) q[3];
rz(-1.3272369870172136) q[3];
ry(0.19772680069633886) q[4];
rz(-0.9606386955460056) q[4];
ry(-0.528011544937751) q[5];
rz(-2.467889539335794) q[5];
ry(0.6617109149344557) q[6];
rz(0.9471624613648548) q[6];
ry(1.3472744671870496) q[7];
rz(2.3410744594694477) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.7134954822262458) q[0];
rz(-1.4499005234333895) q[0];
ry(-1.088399868721221) q[1];
rz(-2.218341506022684) q[1];
ry(-1.1190229226900335) q[2];
rz(0.38289449752967347) q[2];
ry(2.612910039645323) q[3];
rz(-1.120857459708665) q[3];
ry(1.6454496444153197) q[4];
rz(2.8577890526857415) q[4];
ry(-0.8275790545000382) q[5];
rz(-1.881242065700052) q[5];
ry(-1.4961548281055723) q[6];
rz(-0.14218458459468358) q[6];
ry(-0.8350697436362902) q[7];
rz(-2.9142439642174054) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.1833206504918887) q[0];
rz(1.7648814500437628) q[0];
ry(2.6736929440672372) q[1];
rz(-1.382448811304878) q[1];
ry(1.2510187061544464) q[2];
rz(3.012699046381257) q[2];
ry(2.6993041321912052) q[3];
rz(-1.4144829464388782) q[3];
ry(-0.914876185777462) q[4];
rz(-1.3765919751182174) q[4];
ry(2.975748701389707) q[5];
rz(-0.7053897783896728) q[5];
ry(-1.156454176415712) q[6];
rz(-0.6686506366113985) q[6];
ry(2.36544411258555) q[7];
rz(-2.9789911523683235) q[7];