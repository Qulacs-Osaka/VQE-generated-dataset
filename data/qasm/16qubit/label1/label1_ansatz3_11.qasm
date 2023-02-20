OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(0.16071440070084717) q[0];
rz(1.2684992918151714) q[0];
ry(-1.1812900688346817) q[1];
rz(-0.8616135480442457) q[1];
ry(-0.01677379424476566) q[2];
rz(1.2873473240746378) q[2];
ry(-2.9637394869208817) q[3];
rz(-1.0666791155775956) q[3];
ry(-1.9184643709261788) q[4];
rz(-0.7785285924246439) q[4];
ry(-2.5781573203090673) q[5];
rz(0.3199703036576463) q[5];
ry(1.9242366496903527) q[6];
rz(-2.2742137122675725) q[6];
ry(0.009430734059621674) q[7];
rz(2.5211403893276607) q[7];
ry(3.1050543065741594) q[8];
rz(-2.044967047781744) q[8];
ry(-0.514359481508568) q[9];
rz(-1.3327986253418913) q[9];
ry(0.1425486629319646) q[10];
rz(-1.215918961366824) q[10];
ry(1.053471180357552) q[11];
rz(-2.172606871336318) q[11];
ry(2.8621695244969567) q[12];
rz(2.9152022529657433) q[12];
ry(0.002750798758193618) q[13];
rz(-2.4934946896560226) q[13];
ry(-3.1018323892056308) q[14];
rz(2.5415157866918823) q[14];
ry(-1.3621130780990105) q[15];
rz(-0.38679764989398113) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.27433931549962) q[0];
rz(-0.5255568529151731) q[0];
ry(1.4649927309803075) q[1];
rz(0.9958241921559511) q[1];
ry(-2.8341391108416887) q[2];
rz(-1.1351581118712433) q[2];
ry(3.1358695915794277) q[3];
rz(0.12548176893186475) q[3];
ry(1.204078427855701) q[4];
rz(2.296478392005431) q[4];
ry(0.6041898106236048) q[5];
rz(1.6909714191442806) q[5];
ry(2.46790469270533) q[6];
rz(0.8087700042669712) q[6];
ry(0.010699673917298647) q[7];
rz(-0.807252252224711) q[7];
ry(-0.42438129969929506) q[8];
rz(-2.7630150448184603) q[8];
ry(2.216856943751157) q[9];
rz(2.7832199589663587) q[9];
ry(-0.3931566749760407) q[10];
rz(-1.6951776601429183) q[10];
ry(2.954157198436075) q[11];
rz(-2.906129444133824) q[11];
ry(0.23975739609201402) q[12];
rz(-2.4616192210488776) q[12];
ry(0.12398032371650736) q[13];
rz(-2.543549654289225) q[13];
ry(3.1395549211310767) q[14];
rz(2.834828659942229) q[14];
ry(2.4036204204901104) q[15];
rz(1.5194562504395295) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.258714062965475) q[0];
rz(-1.8662604425248717) q[0];
ry(-2.448781128035876) q[1];
rz(-0.4892715507388128) q[1];
ry(-1.0869098846832552) q[2];
rz(-0.24999779604215802) q[2];
ry(-1.5345848373781807) q[3];
rz(-0.4772460945657997) q[3];
ry(-2.560486473505392) q[4];
rz(-1.1609468135249579) q[4];
ry(-0.29774684977933574) q[5];
rz(-2.672123995695203) q[5];
ry(3.0512156773672006) q[6];
rz(-0.5596764713527761) q[6];
ry(-1.2079701097231652) q[7];
rz(1.4607389764000294) q[7];
ry(2.717972959937762) q[8];
rz(2.0723553167097597) q[8];
ry(-1.4492303875092378) q[9];
rz(-1.1137945968975846) q[9];
ry(-1.3778267716223362) q[10];
rz(-1.256117803024087) q[10];
ry(0.06302863197191755) q[11];
rz(2.9830261283351502) q[11];
ry(3.011999809524514) q[12];
rz(1.9847061346394654) q[12];
ry(-0.002104719916231874) q[13];
rz(-0.35599323748245837) q[13];
ry(-3.1017470606061304) q[14];
rz(-2.2274914649380766) q[14];
ry(0.06959400037636243) q[15];
rz(0.4712124703287275) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.824690034497116) q[0];
rz(2.5035933468570937) q[0];
ry(-1.096707120017565) q[1];
rz(-0.2162525521853868) q[1];
ry(2.3386631310026327) q[2];
rz(-2.0440510207490723) q[2];
ry(3.098906205974987) q[3];
rz(-1.0839539033452104) q[3];
ry(-0.023927988547789044) q[4];
rz(-1.1786267810320794) q[4];
ry(-3.1415181129812138) q[5];
rz(0.16069951785220032) q[5];
ry(3.134558194703742) q[6];
rz(2.72211539206552) q[6];
ry(0.04175348833424852) q[7];
rz(-0.1579053514367914) q[7];
ry(-0.22077923375811057) q[8];
rz(-1.9063966042939875) q[8];
ry(-1.1863700990408779) q[9];
rz(-1.4255842622457564) q[9];
ry(-1.1262483854092205) q[10];
rz(3.085352957618441) q[10];
ry(0.769382518151067) q[11];
rz(0.466618800932561) q[11];
ry(1.37611789429565) q[12];
rz(2.6537189502087397) q[12];
ry(-0.1285368549518262) q[13];
rz(2.141374731486598) q[13];
ry(-0.46140709988189466) q[14];
rz(2.1135268351709886) q[14];
ry(2.413066127257076) q[15];
rz(-0.7265342636149649) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.745148779590429) q[0];
rz(2.125858016694719) q[0];
ry(0.638476329213365) q[1];
rz(0.12616289700164263) q[1];
ry(1.9837527894489468) q[2];
rz(-1.7786963449061481) q[2];
ry(-0.22845709017009508) q[3];
rz(3.135811032519724) q[3];
ry(2.3056794182619083) q[4];
rz(2.09046813002948) q[4];
ry(0.8198026622730081) q[5];
rz(-0.8810389601070546) q[5];
ry(-3.0531312607527417) q[6];
rz(2.985882949056121) q[6];
ry(-1.080955431566691) q[7];
rz(1.1948786486594223) q[7];
ry(1.4124465706358165) q[8];
rz(-1.1412127873455624) q[8];
ry(1.3830721497948897) q[9];
rz(-1.7826745787846792) q[9];
ry(2.9710405043606696) q[10];
rz(3.0844612301194334) q[10];
ry(0.23148730834168776) q[11];
rz(-2.601029348049458) q[11];
ry(3.140097471824128) q[12];
rz(2.2524095164210545) q[12];
ry(-3.125819665718658) q[13];
rz(0.789057658353962) q[13];
ry(-0.03381242453952903) q[14];
rz(-2.8867562561505453) q[14];
ry(-0.22664219495003834) q[15];
rz(0.38669342592998834) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.382509252058146) q[0];
rz(2.3352215769660987) q[0];
ry(-1.6882878266937198) q[1];
rz(-2.794448123220511) q[1];
ry(2.427507954674295) q[2];
rz(1.6793989443823358) q[2];
ry(-3.13267108307591) q[3];
rz(-2.8663742236022554) q[3];
ry(-0.018831278841774157) q[4];
rz(0.9358272928887832) q[4];
ry(-0.0007660725138034508) q[5];
rz(0.7327965758042666) q[5];
ry(0.006950334369705935) q[6];
rz(0.8998014832073905) q[6];
ry(0.009605502163030797) q[7];
rz(-0.7781182863053763) q[7];
ry(0.23209408629150355) q[8];
rz(-0.8702002122649918) q[8];
ry(-1.7631379047602322) q[9];
rz(1.5309986492944985) q[9];
ry(-1.7263427934045978) q[10];
rz(-2.82505888772053) q[10];
ry(-2.080423754314255) q[11];
rz(-0.04453969557819004) q[11];
ry(-1.1917544097240684) q[12];
rz(2.778635495402274) q[12];
ry(-0.01154028404608365) q[13];
rz(2.581290425569223) q[13];
ry(2.903868998060613) q[14];
rz(-2.5484402726427966) q[14];
ry(-1.984638709812664) q[15];
rz(-1.7755647111135557) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.9423795822145653) q[0];
rz(-1.6063237287486127) q[0];
ry(0.9288189039943127) q[1];
rz(0.6668935753602595) q[1];
ry(-1.5768283263286063) q[2];
rz(2.1147194768020556) q[2];
ry(0.6695197838893296) q[3];
rz(1.3948210079614778) q[3];
ry(1.0626492750524419) q[4];
rz(-2.0110915178747395) q[4];
ry(1.5731217496784293) q[5];
rz(1.4676556409770765) q[5];
ry(-3.0705392334279864) q[6];
rz(-2.943154071360679) q[6];
ry(0.34214346069258905) q[7];
rz(0.31104622055495834) q[7];
ry(0.2583631298783844) q[8];
rz(-2.7725182670554145) q[8];
ry(0.9811124321332318) q[9];
rz(-0.7072796748602412) q[9];
ry(-0.07533415757054242) q[10];
rz(2.799064814886701) q[10];
ry(0.6341265230797851) q[11];
rz(-1.4861753910421773) q[11];
ry(3.122354608658753) q[12];
rz(-2.831521744113061) q[12];
ry(-0.003771884693953176) q[13];
rz(-2.4383002557370865) q[13];
ry(1.6156443162345069) q[14];
rz(1.9369376522022899) q[14];
ry(1.0710705328828303) q[15];
rz(0.7085533458869103) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-2.036780494095977) q[0];
rz(-2.229272575977226) q[0];
ry(0.8110176159190434) q[1];
rz(2.005349676244319) q[1];
ry(2.1463008613887604) q[2];
rz(0.9088702320663041) q[2];
ry(0.0921479469750972) q[3];
rz(-3.0170006271850576) q[3];
ry(-3.120345234944065) q[4];
rz(2.4512198706033015) q[4];
ry(3.1414840599089144) q[5];
rz(2.0387464112501332) q[5];
ry(0.002082151302000106) q[6];
rz(0.27418965780295773) q[6];
ry(3.1067177686147494) q[7];
rz(-2.806871022204696) q[7];
ry(2.7058302202496796) q[8];
rz(3.037136936347252) q[8];
ry(1.0397225606980747) q[9];
rz(0.5514898761671079) q[9];
ry(-1.585379404776246) q[10];
rz(-0.39809013978594837) q[10];
ry(3.122380290137552) q[11];
rz(1.4200357697031225) q[11];
ry(-3.091813231392379) q[12];
rz(0.8392851510735474) q[12];
ry(-0.7700548692565281) q[13];
rz(1.4160103276496472) q[13];
ry(0.7362669678240263) q[14];
rz(-0.41293733461590243) q[14];
ry(2.5313333445496204) q[15];
rz(-0.6066387449635879) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.681456146659305) q[0];
rz(2.738567035176779) q[0];
ry(-1.1876709899721256) q[1];
rz(-2.1500065286278964) q[1];
ry(1.3436424657376678) q[2];
rz(0.09748530961581024) q[2];
ry(1.1307409233493035) q[3];
rz(-2.344600690257731) q[3];
ry(-0.22490802839631563) q[4];
rz(1.473684634388049) q[4];
ry(-0.5792666841719953) q[5];
rz(3.048950969996769) q[5];
ry(-1.3039302794815448) q[6];
rz(-0.8803564744217801) q[6];
ry(-1.8147547989672823) q[7];
rz(-2.350190072116509) q[7];
ry(2.512279554872141) q[8];
rz(-2.547997799503791) q[8];
ry(1.355538516561758) q[9];
rz(-1.1203539388098176) q[9];
ry(-0.4210448227248796) q[10];
rz(-1.0261148134482196) q[10];
ry(0.0036497444876522067) q[11];
rz(-1.3816029063863318) q[11];
ry(0.6644851655296261) q[12];
rz(-1.6838591066539084) q[12];
ry(3.140436184681369) q[13];
rz(0.718177960733536) q[13];
ry(1.7329882377922805) q[14];
rz(2.161227284473348) q[14];
ry(-2.797136763644861) q[15];
rz(2.057386314559168) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.15724728511736608) q[0];
rz(-2.3898530542846363) q[0];
ry(-2.258581296103527) q[1];
rz(-3.0708139941336574) q[1];
ry(-1.2971673070735974) q[2];
rz(0.6023273921863499) q[2];
ry(0.08538512524224347) q[3];
rz(-1.0648936346310638) q[3];
ry(-3.106946023646937) q[4];
rz(2.8113954295193677) q[4];
ry(3.1383453887914503) q[5];
rz(-0.2244620693931229) q[5];
ry(0.02078863538361464) q[6];
rz(1.7706894041590795) q[6];
ry(-3.1066129064810943) q[7];
rz(2.620134577195986) q[7];
ry(3.1138463585878364) q[8];
rz(-2.7189853683072935) q[8];
ry(-1.4567896102227271) q[9];
rz(2.7319285111753446) q[9];
ry(3.1231368720782076) q[10];
rz(0.5332370943586602) q[10];
ry(-1.587906738935298) q[11];
rz(-0.9965484774888775) q[11];
ry(3.06229962999762) q[12];
rz(1.1328654050614082) q[12];
ry(1.1018347861267834) q[13];
rz(1.6277859666856598) q[13];
ry(1.8364183243205838) q[14];
rz(-1.9604541127906372) q[14];
ry(-0.48900590520622206) q[15];
rz(0.19134460972603315) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.5756404548972754) q[0];
rz(-1.9346779148922009) q[0];
ry(-2.6065507609042946) q[1];
rz(-1.5200487114333372) q[1];
ry(0.53053480202085) q[2];
rz(0.01774053744105965) q[2];
ry(-0.20655797558027444) q[3];
rz(-2.67570121904584) q[3];
ry(0.18848290750273655) q[4];
rz(-0.4754642055174038) q[4];
ry(-2.4753816080703053) q[5];
rz(1.5627714781449522) q[5];
ry(-1.6577924663988821) q[6];
rz(1.9102119761522525) q[6];
ry(1.5149988602153903) q[7];
rz(-1.4497063327941422) q[7];
ry(-1.5959570999092032) q[8];
rz(1.2826965654761655) q[8];
ry(2.229910390806266) q[9];
rz(3.0253098561091405) q[9];
ry(2.6401314236795645) q[10];
rz(-1.9521248544747294) q[10];
ry(0.0024932487505546774) q[11];
rz(1.7647559872135203) q[11];
ry(0.23502940574346454) q[12];
rz(-0.6210835687267657) q[12];
ry(-0.02148342642286625) q[13];
rz(-2.8090791447282517) q[13];
ry(1.6022766808639801) q[14];
rz(1.2177521807063185) q[14];
ry(1.6195778061605397) q[15];
rz(-0.15839792522076665) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.9956391777067712) q[0];
rz(0.244411736329273) q[0];
ry(-1.8473103882306054) q[1];
rz(0.7942528046507124) q[1];
ry(1.2945735902851936) q[2];
rz(-1.8451374966461433) q[2];
ry(0.027129270142587817) q[3];
rz(-2.6629345035655048) q[3];
ry(0.037576240953691276) q[4];
rz(2.977448228380936) q[4];
ry(3.0046083799286665) q[5];
rz(-1.0462936500722764) q[5];
ry(-0.026838014517030957) q[6];
rz(1.995228220799899) q[6];
ry(0.0020387400499242148) q[7];
rz(-3.1132335157844357) q[7];
ry(-3.137464174057618) q[8];
rz(-2.5981625046398853) q[8];
ry(-1.3312978601660876) q[9];
rz(0.8058302544809887) q[9];
ry(3.097724560905894) q[10];
rz(1.9609213313210816) q[10];
ry(-2.039400695886717) q[11];
rz(-2.321836148998036) q[11];
ry(3.0862708814041353) q[12];
rz(-2.6996384849466155) q[12];
ry(-1.184004867854291) q[13];
rz(2.339914184032757) q[13];
ry(3.000846116622665) q[14];
rz(-1.3428654900339954) q[14];
ry(1.7969636636814075) q[15];
rz(1.5361115803123024) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(3.069417583878333) q[0];
rz(0.8700658310719369) q[0];
ry(2.2518870921895706) q[1];
rz(-1.6103547534609115) q[1];
ry(1.1078331480870958) q[2];
rz(2.2839972426451145) q[2];
ry(-0.20490622062315084) q[3];
rz(-0.10293445030030403) q[3];
ry(0.41392796523842174) q[4];
rz(0.6909930217498338) q[4];
ry(-2.6433227404747446) q[5];
rz(2.1072768040470233) q[5];
ry(1.685798463799638) q[6];
rz(0.5119702215125695) q[6];
ry(-2.7907562207355805) q[7];
rz(0.5599437734417262) q[7];
ry(-0.8284534050929678) q[8];
rz(0.3778264160387881) q[8];
ry(-1.3568136021750385) q[9];
rz(2.8929255727044914) q[9];
ry(2.7861789170943467) q[10];
rz(-2.4360253415544753) q[10];
ry(-3.1319637019990685) q[11];
rz(1.5135060398073186) q[11];
ry(-0.36998258998715244) q[12];
rz(-2.737437123936817) q[12];
ry(1.5650147915811115) q[13];
rz(1.5700395025370557) q[13];
ry(-1.7074214468167515) q[14];
rz(-1.828299854173574) q[14];
ry(1.3872828991293302) q[15];
rz(-2.663538054402381) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.9809948687946672) q[0];
rz(2.441356796600406) q[0];
ry(-2.792498310728015) q[1];
rz(0.530751943521083) q[1];
ry(-2.8873971378311167) q[2];
rz(2.8459857795646606) q[2];
ry(3.113520613853171) q[3];
rz(1.7854264529086799) q[3];
ry(0.06011866072649586) q[4];
rz(1.6597513574642682) q[4];
ry(0.2117710490810918) q[5];
rz(1.722454193152279) q[5];
ry(-3.135367508900007) q[6];
rz(1.1129320762573942) q[6];
ry(-3.1193422196889475) q[7];
rz(0.40769438266740793) q[7];
ry(-0.07899491376794864) q[8];
rz(1.5184832340275725) q[8];
ry(-2.8467172984691125) q[9];
rz(-0.17884675625956967) q[9];
ry(-3.0811378820494464) q[10];
rz(-1.0721070988095198) q[10];
ry(0.009061439723834263) q[11];
rz(2.6069647157098337) q[11];
ry(3.120509506181964) q[12];
rz(-2.0495107023439765) q[12];
ry(1.5460378019220522) q[13];
rz(0.4446230820730186) q[13];
ry(-1.5143358847726702) q[14];
rz(-0.942809211546306) q[14];
ry(1.667854598239276) q[15];
rz(-0.057947552706191514) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.844133696223574) q[0];
rz(-0.6037935050027485) q[0];
ry(-0.9272103716403032) q[1];
rz(1.4023381755131739) q[1];
ry(1.3588153876289697) q[2];
rz(2.718600062534655) q[2];
ry(-0.7199037488404745) q[3];
rz(2.9960238027073753) q[3];
ry(-1.3960384405892956) q[4];
rz(2.6931858557327506) q[4];
ry(-0.2614900479219005) q[5];
rz(1.5099272838488909) q[5];
ry(1.0086623575540639) q[6];
rz(-1.7716654814827046) q[6];
ry(-2.902067368046609) q[7];
rz(-1.5241402337480254) q[7];
ry(1.7513563227688245) q[8];
rz(-2.485103468187187) q[8];
ry(0.3706161087729113) q[9];
rz(2.1820930109184946) q[9];
ry(2.7415018744079442) q[10];
rz(-0.23233333779490725) q[10];
ry(1.401508664373324) q[11];
rz(0.22220011286050403) q[11];
ry(3.08015995516455) q[12];
rz(-2.4071916422300403) q[12];
ry(-3.057086902518798) q[13];
rz(0.08768579457449022) q[13];
ry(0.11075026192734061) q[14];
rz(-0.027592243608562054) q[14];
ry(-1.5970757656974397) q[15];
rz(0.7423837059136407) q[15];