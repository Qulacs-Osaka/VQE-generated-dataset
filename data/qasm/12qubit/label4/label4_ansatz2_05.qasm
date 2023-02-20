OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.7182921248440354e-06) q[0];
rz(-2.992925939715104) q[0];
ry(1.655106565841689e-06) q[1];
rz(2.1673782500608274) q[1];
ry(-1.5707961125141385) q[2];
rz(-1.5707971198063575) q[2];
ry(1.5708040371826586) q[3];
rz(-2.211656239308075) q[3];
ry(1.5670321804349765) q[4];
rz(-1.5707986507881895) q[4];
ry(-3.141531777330169) q[5];
rz(-1.1138501437675572) q[5];
ry(2.898984437936008) q[6];
rz(-3.141581504856303) q[6];
ry(2.706907893669878) q[7];
rz(-3.1415819429357272) q[7];
ry(0.8253737500991781) q[8];
rz(-1.6267401931045722) q[8];
ry(3.1415880395310425) q[9];
rz(-3.0235758602584566) q[9];
ry(1.515910113146617) q[10];
rz(-9.6502875604677e-07) q[10];
ry(1.5707965782048747) q[11];
rz(-3.1415914631264634) q[11];
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
ry(1.5707957852296737) q[0];
rz(-3.14140023807742) q[0];
ry(1.5707977134892377) q[1];
rz(-1.4428601872542157) q[1];
ry(1.5707940150283042) q[2];
rz(2.9762119538728062) q[2];
ry(-0.00012807671702042) q[3];
rz(-2.500656803374089) q[3];
ry(-1.5707971977802426) q[4];
rz(0.30971530858269) q[4];
ry(1.8367418856365827) q[5];
rz(0.11128065511341925) q[5];
ry(1.5708240925746715) q[6];
rz(-1.5708000089932455) q[6];
ry(1.5707871653476362) q[7];
rz(-1.5708082237091707) q[7];
ry(-3.1415902396031146) q[8];
rz(1.514855658366553) q[8];
ry(-4.24774700213959e-06) q[9];
rz(-2.8280909807315493) q[9];
ry(1.5707967522384125) q[10];
rz(1.5707969112461342) q[10];
ry(1.8614358643673512) q[11];
rz(1.5707967681217507) q[11];
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
ry(-0.20389035816281265) q[0];
rz(0.48287198990787245) q[0];
ry(3.6479391241073245e-07) q[1];
rz(-0.1279360841135455) q[1];
ry(7.848713678981767e-08) q[2];
rz(-1.8074969639655671) q[2];
ry(0.3640099222171056) q[3];
rz(1.5707294684691806) q[3];
ry(3.1366993669028465) q[4];
rz(-2.8965047177344174) q[4];
ry(-3.138711796768641) q[5];
rz(1.6820753613232489) q[5];
ry(-1.5707937276707729) q[6];
rz(-3.0693153851453574) q[6];
ry(1.5707961545518014) q[7];
rz(-2.8664859892047705) q[7];
ry(-1.5707993057550702) q[8];
rz(1.570797185080715) q[8];
ry(-3.141559621083853) q[9];
rz(-2.7560474608661143) q[9];
ry(-1.5708642139298021) q[10];
rz(1.630570705055501) q[10];
ry(-1.206853795376919) q[11];
rz(1.5707959603370927) q[11];
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
ry(3.141592400398873) q[0];
rz(1.3304573012230994) q[0];
ry(-1.5707889760959173) q[1];
rz(2.032569970108711) q[1];
ry(-0.00012667140115407705) q[2];
rz(-2.682561461804628) q[2];
ry(-0.3832966604273764) q[3];
rz(-1.5708079875872052) q[3];
ry(-9.945664982641925e-08) q[4];
rz(-1.0892524022125698) q[4];
ry(-1.5708032488081258) q[5];
rz(-1.5969466174000777) q[5];
ry(-2.0147494850932706) q[6];
rz(1.5916745651746922) q[6];
ry(2.832602853834092) q[7];
rz(0.4183060443659906) q[7];
ry(1.5707925185962788) q[8];
rz(-2.80493521428211) q[8];
ry(-3.1415916074658568) q[9];
rz(2.024554146942208) q[9];
ry(2.8961376757844204) q[10];
rz(-0.11193114423447614) q[10];
ry(2.5709609818559813) q[11];
rz(1.5707959593584104) q[11];
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
ry(3.1415918265338756) q[0];
rz(-0.47433648184195915) q[0];
ry(-1.881636895184613e-06) q[1];
rz(-2.016603222794014) q[1];
ry(3.141589443818068) q[2];
rz(2.599235248749077) q[2];
ry(-2.1654573482126294) q[3];
rz(2.898843416985794) q[3];
ry(-8.813717214542276e-07) q[4];
rz(1.9034313530949547) q[4];
ry(3.141590297710023) q[5];
rz(1.5804388470578494) q[5];
ry(-1.5707955947291383) q[6];
rz(-0.8654528046602172) q[6];
ry(1.5708501558939194) q[7];
rz(-2.1692318644309454) q[7];
ry(4.164181148881478e-06) q[8];
rz(-1.342512134939192) q[8];
ry(1.5707929007739982) q[9];
rz(-0.14460469936241083) q[9];
ry(1.5708128261258434) q[10];
rz(3.1304438212711925) q[10];
ry(-0.9762894299192757) q[11];
rz(1.5707942029061777) q[11];
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
ry(3.260630414203547e-05) q[0];
rz(-0.24907211797937306) q[0];
ry(-3.1392072120108887) q[1];
rz(0.01595931824192132) q[1];
ry(3.956591678361364e-07) q[2];
rz(-1.0354100861812168) q[2];
ry(2.1274469020395313e-06) q[3];
rz(1.8135267357306597) q[3];
ry(3.1413830251645503) q[4];
rz(-2.125252539704703) q[4];
ry(-0.0021897573825370004) q[5];
rz(3.010037502526239) q[5];
ry(-3.14159140666081) q[6];
rz(-1.0089726034134543) q[6];
ry(3.1415920647838007) q[7];
rz(3.0140112848576677) q[7];
ry(-3.141324315753174) q[8];
rz(0.3953306057900677) q[8];
ry(0.8167470630581916) q[9];
rz(1.780355177924533) q[9];
ry(3.141589804502241) q[10];
rz(3.0924028599992233) q[10];
ry(1.5707946546366793) q[11];
rz(-1.5707962869368266) q[11];
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
ry(1.5708142503230966) q[0];
rz(-2.1170391699366324) q[0];
ry(-1.5708063389920337) q[1];
rz(-2.1174522002907716) q[1];
ry(-1.2026680664689595e-06) q[2];
rz(0.04034786088165996) q[2];
ry(1.5707859997185736) q[3];
rz(-1.6498051132206812) q[3];
ry(1.5716006495309935) q[4];
rz(1.0107051952747375) q[4];
ry(0.007522864534882108) q[5];
rz(-2.2421637639236556) q[5];
ry(-1.7641330912532286e-05) q[6];
rz(1.7631255536853514) q[6];
ry(-3.1415908517257667) q[7];
rz(-0.5142514488580412) q[7];
ry(-3.141246953966129) q[8];
rz(-1.4612959876436726) q[8];
ry(-1.5707752617072261) q[9];
rz(1.943488732596526) q[9];
ry(1.7283029520874605e-06) q[10];
rz(-0.8204539506095987) q[10];
ry(0.2166079813475159) q[11];
rz(-1.5707962547441154) q[11];
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
ry(1.5708025916163284) q[0];
rz(-1.5707947946248648) q[0];
ry(-1.5708005903499869) q[1];
rz(-2.6359781882852773) q[1];
ry(1.570797229583766) q[2];
rz(-1.5707977882438406) q[2];
ry(3.1415919071396794) q[3];
rz(-0.2956147113114717) q[3];
ry(-0.00039888636868978144) q[4];
rz(2.1328414146468555) q[4];
ry(0.000986963598790397) q[5];
rz(2.337759103287453) q[5];
ry(1.443097427044921e-06) q[6];
rz(1.9538029181970173) q[6];
ry(-3.1415926402874135) q[7];
rz(2.613207464231202) q[7];
ry(-3.1415861772698603) q[8];
rz(-0.42360346859229503) q[8];
ry(-3.1415426296598077) q[9];
rz(-2.6152028199173705) q[9];
ry(-3.1415925071267314) q[10];
rz(-1.9360389476324293) q[10];
ry(1.5707965293768291) q[11];
rz(-1.570793541971932) q[11];
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
ry(-1.5707957732721978) q[0];
rz(-0.581631429489673) q[0];
ry(2.759246312329996) q[1];
rz(-0.7466244357975453) q[1];
ry(1.5707963364475306) q[2];
rz(1.3274589210011616) q[2];
ry(-1.570796432312945) q[3];
rz(0.3496715636976573) q[3];
ry(-1.5707936046254187) q[4];
rz(1.32745588606768) q[4];
ry(0.1209731106715708) q[5];
rz(0.34983994623329373) q[5];
ry(3.141591723825887) q[6];
rz(1.7592779344898073) q[6];
ry(-3.141592425341852) q[7];
rz(-2.335201974211348) q[7];
ry(-3.141581564375542) q[8];
rz(0.6247092550512416) q[8];
ry(-3.141578417992695) q[9];
rz(-2.6382384299525823) q[9];
ry(-3.1415918820220905) q[10];
rz(-2.89182138098371) q[10];
ry(1.5707967242332845) q[11];
rz(1.9204532501202523) q[11];