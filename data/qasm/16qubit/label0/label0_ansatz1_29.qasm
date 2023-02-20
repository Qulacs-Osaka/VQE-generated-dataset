OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.085946582731755) q[0];
rz(-1.3626204566374218) q[0];
ry(-1.7128381510846893) q[1];
rz(-1.8049341029188812) q[1];
ry(-2.0690282276943117) q[2];
rz(-0.8405395129794297) q[2];
ry(-0.13803604644800505) q[3];
rz(1.283133669718051) q[3];
ry(-3.1386675072585386) q[4];
rz(-2.603946977839901) q[4];
ry(1.506379337440749) q[5];
rz(-1.7673235806772516) q[5];
ry(-0.8587244382782723) q[6];
rz(-1.765848642441898) q[6];
ry(-0.004024140002240814) q[7];
rz(1.0267339420194617) q[7];
ry(-2.54699455887837) q[8];
rz(-0.1003505205787194) q[8];
ry(2.705721450987114) q[9];
rz(-2.630240034940385) q[9];
ry(-0.456626447113726) q[10];
rz(-2.5514725344419578) q[10];
ry(-0.0018158810771294258) q[11];
rz(-0.9880610878974978) q[11];
ry(1.927502778467909) q[12];
rz(-2.3678341091984647) q[12];
ry(0.2434574455904387) q[13];
rz(-1.9265731287854972) q[13];
ry(3.1360064027516374) q[14];
rz(-1.7721586681889612) q[14];
ry(2.2378265445067864) q[15];
rz(-1.1194244699859652) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.7843770393138074) q[0];
rz(-2.81198975839851) q[0];
ry(2.011047172956654) q[1];
rz(-0.14729506408448026) q[1];
ry(3.1082797774524535) q[2];
rz(2.32594256332916) q[2];
ry(-2.918089624327634) q[3];
rz(-1.3355691870093107) q[3];
ry(3.1378072516763984) q[4];
rz(0.48718578321780087) q[4];
ry(1.4701117997511588) q[5];
rz(2.0208225534453055) q[5];
ry(1.094181365801151) q[6];
rz(-1.5155307566512468) q[6];
ry(-3.105099615763741) q[7];
rz(0.6982304478147188) q[7];
ry(0.18606854682348842) q[8];
rz(-0.971502191219475) q[8];
ry(-0.4190718421874289) q[9];
rz(-0.6584558497159403) q[9];
ry(1.4591849347400552) q[10];
rz(-3.0538143059543628) q[10];
ry(3.1414020231106785) q[11];
rz(-1.2247427267196802) q[11];
ry(0.19356778084819812) q[12];
rz(0.5780234886460027) q[12];
ry(2.3980769633822305) q[13];
rz(0.24771313525727662) q[13];
ry(3.136197920067778) q[14];
rz(-1.2770301259434547) q[14];
ry(2.1685212262573907) q[15];
rz(0.6312709670706189) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.5312904979827853) q[0];
rz(-1.2502622547749849) q[0];
ry(-0.3753963436673002) q[1];
rz(-3.1226301647061754) q[1];
ry(0.47117783319951423) q[2];
rz(-2.5136101594924463) q[2];
ry(-3.1038201419558376) q[3];
rz(0.2549272047426978) q[3];
ry(0.5421280819108043) q[4];
rz(-1.7362864558042093) q[4];
ry(-1.2840956579477776) q[5];
rz(1.0783607412212113) q[5];
ry(0.7770773746416388) q[6];
rz(-3.100540066342985) q[6];
ry(-3.1391228203528505) q[7];
rz(2.870471367270536) q[7];
ry(0.7934777947567869) q[8];
rz(2.339511542218265) q[8];
ry(0.07500147105752397) q[9];
rz(0.6116252180219415) q[9];
ry(-2.9965953993341365) q[10];
rz(-0.17593743307414478) q[10];
ry(3.141392013551541) q[11];
rz(1.3818801726165044) q[11];
ry(-1.9000843160114167) q[12];
rz(1.1376953925590403) q[12];
ry(-0.4048409440076331) q[13];
rz(-0.26597552487332843) q[13];
ry(2.4489054812480253) q[14];
rz(1.4105643036498163) q[14];
ry(0.605210700413731) q[15];
rz(2.5621561708307996) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.9004058191865378) q[0];
rz(-2.811617145639935) q[0];
ry(3.0897450174510466) q[1];
rz(2.6214195684451673) q[1];
ry(-0.3360722708784003) q[2];
rz(0.5756152097281189) q[2];
ry(-2.8890289717663173) q[3];
rz(2.4140500248858303) q[3];
ry(-3.1408834240198984) q[4];
rz(1.7216181646607385) q[4];
ry(0.01677309340852711) q[5];
rz(-2.7599924319931826) q[5];
ry(1.5121706641808652) q[6];
rz(-0.017948331837223996) q[6];
ry(2.9191862145642804) q[7];
rz(-0.5187395274968519) q[7];
ry(-2.795874488331483) q[8];
rz(-0.3166218206941562) q[8];
ry(-0.8278711553841438) q[9];
rz(-0.13985433312835252) q[9];
ry(-1.199456705831845) q[10];
rz(0.29309447410528566) q[10];
ry(0.8694610808599931) q[11];
rz(0.6705950131452916) q[11];
ry(1.7943101020039316) q[12];
rz(0.5333488713558729) q[12];
ry(-3.1234623479844252) q[13];
rz(2.2626883224218517) q[13];
ry(-2.864590001493513) q[14];
rz(-1.6136324331824623) q[14];
ry(3.010082625347698) q[15];
rz(-2.682886297802075) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.1087875555303177) q[0];
rz(0.6273230512811375) q[0];
ry(1.628644312157511) q[1];
rz(-2.2226812942305108) q[1];
ry(-0.43800480889556237) q[2];
rz(-0.9925136663839362) q[2];
ry(-0.6066876528922789) q[3];
rz(2.2820193040950363) q[3];
ry(1.5067401464242425) q[4];
rz(0.535549320350847) q[4];
ry(-1.1295575913050309) q[5];
rz(2.976702674089979) q[5];
ry(2.6512901279323895) q[6];
rz(-3.1136205594907937) q[6];
ry(3.1397764937560693) q[7];
rz(1.367507825738104) q[7];
ry(2.9731184920407587) q[8];
rz(0.4543271436976683) q[8];
ry(-2.353122256784574) q[9];
rz(-3.140393942619857) q[9];
ry(-2.4986852586713666) q[10];
rz(3.0053905405890675) q[10];
ry(1.2351691646290714) q[11];
rz(2.388111726080204) q[11];
ry(-3.1400874625667745) q[12];
rz(-0.9005822710486949) q[12];
ry(1.6091791570696377) q[13];
rz(0.8936924298177341) q[13];
ry(2.434704768085764) q[14];
rz(-1.7508335432751256) q[14];
ry(2.6392224620663693) q[15];
rz(-0.1782207790432681) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.8338646423951726) q[0];
rz(-0.18614108395259849) q[0];
ry(3.116364997002288) q[1];
rz(1.2460820702015578) q[1];
ry(-3.05071425330305) q[2];
rz(-1.5064741602609792) q[2];
ry(-2.8173551193215234) q[3];
rz(-0.3052616673340457) q[3];
ry(-0.027799430849187345) q[4];
rz(-0.5790793828875689) q[4];
ry(0.024465284658631813) q[5];
rz(2.4652927966427276) q[5];
ry(-2.294811751790399) q[6];
rz(-0.0005987357646688451) q[6];
ry(-0.08008589777139274) q[7];
rz(-2.6117923839102013) q[7];
ry(1.4688024997042053) q[8];
rz(2.7773778437817684) q[8];
ry(-0.03930698177722469) q[9];
rz(-2.4890142983170653) q[9];
ry(-0.47883304869510734) q[10];
rz(-3.071907659030069) q[10];
ry(0.0034732854312062855) q[11];
rz(-2.8784166300341134) q[11];
ry(-0.004405491679386309) q[12];
rz(3.0123066437973494) q[12];
ry(2.1891940105996337) q[13];
rz(3.1328596124381365) q[13];
ry(0.36271113665020227) q[14];
rz(2.602451999520884) q[14];
ry(1.956721904877397) q[15];
rz(-2.3460718491824055) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.467465214471649) q[0];
rz(1.8440835429875717) q[0];
ry(-0.4378729898644008) q[1];
rz(1.0101481901154061) q[1];
ry(2.9221150387451766) q[2];
rz(-1.497762495664464) q[2];
ry(-2.398149399960391) q[3];
rz(2.3736167630730565) q[3];
ry(2.7592437646783545) q[4];
rz(-1.8925262494776405) q[4];
ry(-1.7928355772827402) q[5];
rz(-0.7844803984167951) q[5];
ry(0.31128183600433845) q[6];
rz(-0.40314199365016185) q[6];
ry(3.1371042748630598) q[7];
rz(-2.55648277993881) q[7];
ry(2.916876636477879) q[8];
rz(-1.1164810546890198) q[8];
ry(-0.2967703568094562) q[9];
rz(-3.0055876561103023) q[9];
ry(2.424338606454653) q[10];
rz(2.1738647812072873) q[10];
ry(2.260394900752458) q[11];
rz(2.3278194083674073) q[11];
ry(-1.4988661666232614) q[12];
rz(2.7079021329616735) q[12];
ry(0.9775024966973765) q[13];
rz(-1.9269126545680253) q[13];
ry(-2.99235239044506) q[14];
rz(3.135639988715408) q[14];
ry(-2.858116303832774) q[15];
rz(0.9429812631160797) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.9228418663117068) q[0];
rz(-2.0378048101139683) q[0];
ry(-3.123382341620023) q[1];
rz(-2.1115541528325683) q[1];
ry(-0.030429210338442484) q[2];
rz(0.8504611277689067) q[2];
ry(-1.6693116287007763) q[3];
rz(1.036275007649593) q[3];
ry(-3.1126349044824457) q[4];
rz(-1.0471623123669138) q[4];
ry(-0.0049906131953088104) q[5];
rz(-0.7284262044962578) q[5];
ry(0.009849318384341523) q[6];
rz(2.6561628702953044) q[6];
ry(0.09132921744491596) q[7];
rz(-2.961547033008196) q[7];
ry(-0.6646081006919908) q[8];
rz(0.6675427336226433) q[8];
ry(-1.7026634724334) q[9];
rz(2.304395157384068) q[9];
ry(-0.20869966063472134) q[10];
rz(-2.585265348411148) q[10];
ry(-0.0022053445569669705) q[11];
rz(2.1343136363882707) q[11];
ry(0.004868587237540467) q[12];
rz(0.8351054475568562) q[12];
ry(2.654213198024824) q[13];
rz(2.8704518048985532) q[13];
ry(-0.7557113626509013) q[14];
rz(-2.4885242665923033) q[14];
ry(-1.646016287560839) q[15];
rz(-2.843278557534543) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.6354180074145868) q[0];
rz(-0.15985012262825227) q[0];
ry(0.052047644120460035) q[1];
rz(-2.884405993825052) q[1];
ry(-0.05942219267025717) q[2];
rz(-1.7266598742577752) q[2];
ry(-0.7496631339410893) q[3];
rz(-2.4078306459073366) q[3];
ry(-2.9377446958613156) q[4];
rz(1.9374140864870004) q[4];
ry(1.1022766465530744) q[5];
rz(-2.068027408000124) q[5];
ry(0.19028957792172152) q[6];
rz(1.0918232320282852) q[6];
ry(0.004047168611370331) q[7];
rz(-0.8540242111184776) q[7];
ry(-3.0721656656001137) q[8];
rz(-0.40107058221241143) q[8];
ry(3.1369462754226776) q[9];
rz(-1.6964734278056328) q[9];
ry(-3.0237226990371724) q[10];
rz(-0.9744529525725353) q[10];
ry(-0.9114825937940578) q[11];
rz(-0.12269201054070501) q[11];
ry(-1.9096355137829533) q[12];
rz(-0.6391728240176682) q[12];
ry(2.2013654110891503) q[13];
rz(1.4883276023015921) q[13];
ry(0.3907263943020993) q[14];
rz(0.6273978368441303) q[14];
ry(-0.990899571638189) q[15];
rz(-2.1951719411039594) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.9316862592503886) q[0];
rz(1.081395773670859) q[0];
ry(3.120920600565952) q[1];
rz(-1.4705444229841373) q[1];
ry(1.8313899353149905) q[2];
rz(-0.1324114920379449) q[2];
ry(2.739754528948951) q[3];
rz(2.441271555300704) q[3];
ry(0.03891216946119336) q[4];
rz(0.12409108586471264) q[4];
ry(3.13411140088733) q[5];
rz(2.6656886074702504) q[5];
ry(-0.3651279893724143) q[6];
rz(-1.8356344846893977) q[6];
ry(-2.892498002926602) q[7];
rz(-1.626618017939303) q[7];
ry(-1.1322167842680955) q[8];
rz(1.9125274090907105) q[8];
ry(-1.42844379076109) q[9];
rz(2.8595093429375766) q[9];
ry(2.1506117483591796) q[10];
rz(-2.3490744004469635) q[10];
ry(3.096308889431047) q[11];
rz(0.7323636493284925) q[11];
ry(-0.9633969457621311) q[12];
rz(0.975202624789401) q[12];
ry(1.5039630021535633) q[13];
rz(0.06297249421241347) q[13];
ry(-1.9062754667444999) q[14];
rz(-1.016063476259828) q[14];
ry(3.1189209384896612) q[15];
rz(-2.8354868901387924) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.5420405526171623) q[0];
rz(0.5742814228765677) q[0];
ry(-3.0551375518001724) q[1];
rz(-1.584541602751755) q[1];
ry(-0.8095087990573084) q[2];
rz(-3.069883285139793) q[2];
ry(0.19848726592422586) q[3];
rz(-3.0078170566835003) q[3];
ry(-2.3625532107872402) q[4];
rz(-2.7614510246593) q[4];
ry(2.2089661397635414) q[5];
rz(-1.7567326756533834) q[5];
ry(2.5803482429234434) q[6];
rz(-2.035273535827618) q[6];
ry(-2.955073016333117) q[7];
rz(-1.4101922193477927) q[7];
ry(-1.4656300266158842) q[8];
rz(-0.6190752638541229) q[8];
ry(-1.7444440715531782) q[9];
rz(1.6792123078071495) q[9];
ry(2.5024827729292802) q[10];
rz(3.097005641998564) q[10];
ry(-3.129752150064256) q[11];
rz(-2.6841020665158397) q[11];
ry(3.11769777253967) q[12];
rz(1.822442157997921) q[12];
ry(3.1409841361865545) q[13];
rz(-2.229592604599584) q[13];
ry(2.6805840609868) q[14];
rz(1.715764095158588) q[14];
ry(-3.139046992222596) q[15];
rz(-0.7546188255252282) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.0565687448310173) q[0];
rz(0.3673159006688458) q[0];
ry(3.1259894371663752) q[1];
rz(-2.886593448848687) q[1];
ry(-1.7746746260437742) q[2];
rz(-2.4081718072042233) q[2];
ry(-1.0983368282648633) q[3];
rz(0.9230395725147861) q[3];
ry(2.8902919894044197) q[4];
rz(1.8970125255527472) q[4];
ry(-0.03527741400528317) q[5];
rz(-1.2179461571716514) q[5];
ry(0.1110399730942957) q[6];
rz(-2.729683036368176) q[6];
ry(0.15085844784472632) q[7];
rz(0.8165689359992907) q[7];
ry(0.007249882884362791) q[8];
rz(1.0608587756931889) q[8];
ry(-0.006751949391733303) q[9];
rz(2.5549003144015234) q[9];
ry(1.556265812042487) q[10];
rz(0.18700992650596682) q[10];
ry(-0.02241611808620008) q[11];
rz(0.5791162048588702) q[11];
ry(-0.2998028572233657) q[12];
rz(2.5998789392288746) q[12];
ry(-0.4361805813117127) q[13];
rz(-2.480480468278894) q[13];
ry(1.2016203706945987) q[14];
rz(2.7749411412001552) q[14];
ry(-1.8022947481399805) q[15];
rz(2.6921563603993617) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.5360356739373994) q[0];
rz(2.8085403177666026) q[0];
ry(-0.7134456178518276) q[1];
rz(-0.0018756823892820407) q[1];
ry(0.7180614129846707) q[2];
rz(-3.0339327853276865) q[2];
ry(2.8732844024395505) q[3];
rz(1.2765402539044377) q[3];
ry(0.19082658749825884) q[4];
rz(2.1809527457300026) q[4];
ry(-3.014415562592695) q[5];
rz(0.25820811891075757) q[5];
ry(0.24391154279668142) q[6];
rz(-0.8602359366058607) q[6];
ry(0.6884567752414466) q[7];
rz(-3.0128747763986294) q[7];
ry(1.2254667546512898) q[8];
rz(-1.9224670718162455) q[8];
ry(1.3542400901247538) q[9];
rz(-1.5117131527507368) q[9];
ry(-2.65743170452266) q[10];
rz(-2.0752880037200634) q[10];
ry(-0.0005696937433663507) q[11];
rz(-1.8036486958008533) q[11];
ry(-3.0510805501199827) q[12];
rz(-3.0539989283902353) q[12];
ry(0.002884739038223394) q[13];
rz(0.9773028232767275) q[13];
ry(0.6715168614591347) q[14];
rz(0.4595296449105046) q[14];
ry(-0.8260234878861243) q[15];
rz(0.6801250513082974) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.5498590288224473) q[0];
rz(1.019242438282861) q[0];
ry(-0.03369771983703417) q[1];
rz(2.5428543975331745) q[1];
ry(3.111897794802177) q[2];
rz(1.0972228614998185) q[2];
ry(-3.0680208734209677) q[3];
rz(-2.023381413372432) q[3];
ry(-1.0197629990737642) q[4];
rz(0.13740951304139326) q[4];
ry(0.002475860981057565) q[5];
rz(-1.8924762381095233) q[5];
ry(-2.6488472563769205) q[6];
rz(0.29728434080595445) q[6];
ry(-0.38153952338740194) q[7];
rz(2.6188935246420706) q[7];
ry(0.003552085939492713) q[8];
rz(0.48444862445358705) q[8];
ry(-3.1411106840388374) q[9];
rz(0.6436931625717923) q[9];
ry(-1.5285588119604023) q[10];
rz(-1.3845244079821497) q[10];
ry(-3.134536884202008) q[11];
rz(-0.822320905412282) q[11];
ry(-1.5326695153812235) q[12];
rz(0.051673933911464694) q[12];
ry(2.0528430239072595) q[13];
rz(2.3151271090396532) q[13];
ry(2.3937402856186107) q[14];
rz(0.835444350148653) q[14];
ry(2.8999685845635192) q[15];
rz(1.6462226879309645) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.5544538036304756) q[0];
rz(1.934485560078321) q[0];
ry(2.2364557508320404) q[1];
rz(0.48206163448865635) q[1];
ry(1.8025425960562222) q[2];
rz(-2.9767796856427697) q[2];
ry(-0.15392362316659103) q[3];
rz(2.4632267337725375) q[3];
ry(-0.250179353567483) q[4];
rz(-1.2276917733928947) q[4];
ry(2.013670356801493) q[5];
rz(-1.3170075048031586) q[5];
ry(1.9014129916405338) q[6];
rz(1.821591177943274) q[6];
ry(2.244634361025879) q[7];
rz(-1.600431083696848) q[7];
ry(-2.00012973310656) q[8];
rz(2.5382605419196835) q[8];
ry(2.4550935219714938) q[9];
rz(3.017842248091338) q[9];
ry(0.8887020909807857) q[10];
rz(-1.2544321891236905) q[10];
ry(3.053991980164462) q[11];
rz(-2.1548462352359614) q[11];
ry(-1.638399973883296) q[12];
rz(3.0090465798297776) q[12];
ry(-0.0008486882939570292) q[13];
rz(-0.7934417156822353) q[13];
ry(-1.9483490721191687) q[14];
rz(-1.820030965745425) q[14];
ry(-2.971674770934697) q[15];
rz(0.8227418533149263) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.10316642603515656) q[0];
rz(2.03990482170191) q[0];
ry(0.03905544859842892) q[1];
rz(-2.3356845614688098) q[1];
ry(-3.127679508705316) q[2];
rz(1.5071069691353867) q[2];
ry(-2.2653642916445125) q[3];
rz(-2.0310860351667084) q[3];
ry(0.140243175805578) q[4];
rz(0.15378536634558393) q[4];
ry(-3.0783675874403276) q[5];
rz(-0.6607691129957667) q[5];
ry(-0.03500749691456306) q[6];
rz(0.754466691171122) q[6];
ry(0.003520487490509967) q[7];
rz(0.9936551604724251) q[7];
ry(-3.132240324733571) q[8];
rz(-1.3904018872044173) q[8];
ry(2.864985205320887) q[9];
rz(0.6175366189048077) q[9];
ry(2.0614148965434596) q[10];
rz(1.0431486891577126) q[10];
ry(-1.4039551202781893) q[11];
rz(3.132773213549671) q[11];
ry(2.6986332814136538) q[12];
rz(-0.1649154281541403) q[12];
ry(3.086031689449352) q[13];
rz(-1.37148818104437) q[13];
ry(-1.5776383268779597) q[14];
rz(-2.4983151117079676) q[14];
ry(-0.18940947708947234) q[15];
rz(1.3665931346583937) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.3399406008219197) q[0];
rz(2.1099947329652986) q[0];
ry(-1.283450430765777) q[1];
rz(1.5724675938714463) q[1];
ry(2.618259981360198) q[2];
rz(0.7311958866643907) q[2];
ry(-2.7679982589896484) q[3];
rz(-1.2918294218060176) q[3];
ry(-0.39866501362354567) q[4];
rz(-0.7469561908310647) q[4];
ry(-2.1735970722854163) q[5];
rz(2.731564320860765) q[5];
ry(-2.0441261898985252) q[6];
rz(-2.9272962829660103) q[6];
ry(-3.0932875219597777) q[7];
rz(0.658575805268562) q[7];
ry(2.5750154270219077) q[8];
rz(-0.012527769914080444) q[8];
ry(2.448817812863877) q[9];
rz(-2.229717696335759) q[9];
ry(0.005417392803407056) q[10];
rz(1.2981340219047368) q[10];
ry(1.5360138024710206) q[11];
rz(3.1277466528628866) q[11];
ry(0.6944858053407968) q[12];
rz(-0.6046679941391819) q[12];
ry(-0.588933363348733) q[13];
rz(-1.0078317713922345) q[13];
ry(2.2421696718687607) q[14];
rz(-1.3895324842348904) q[14];
ry(-1.1135546856741243) q[15];
rz(0.009886668717794755) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.1718572204250695) q[0];
rz(-2.0748063496843434) q[0];
ry(3.1340646344682077) q[1];
rz(-1.8411867883039543) q[1];
ry(0.03213290229314402) q[2];
rz(0.030791013706511033) q[2];
ry(0.024721931116194362) q[3];
rz(-0.27848604251448794) q[3];
ry(2.3993329458999986) q[4];
rz(0.6517478332092513) q[4];
ry(3.083297272556837) q[5];
rz(-2.529926570548698) q[5];
ry(2.993530295853517) q[6];
rz(1.3385004953280926) q[6];
ry(2.9895349464465633) q[7];
rz(0.6416291340581024) q[7];
ry(-0.009963696514510012) q[8];
rz(-2.8969688935171245) q[8];
ry(-0.07406800441279149) q[9];
rz(-0.6358435249802578) q[9];
ry(-2.8085737799252453) q[10];
rz(-2.1355438530020256) q[10];
ry(-1.1291585834692703) q[11];
rz(3.1212480237505784) q[11];
ry(-3.141472998103762) q[12];
rz(-0.5997804574132564) q[12];
ry(-0.000937578883251966) q[13];
rz(-0.868838705219324) q[13];
ry(-0.006810837036939077) q[14];
rz(0.6728273813471253) q[14];
ry(0.8011602443852333) q[15];
rz(-1.0707256025385774) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.4304652676630276) q[0];
rz(-1.4279846595599324) q[0];
ry(-0.13004805574867628) q[1];
rz(-2.405813137266703) q[1];
ry(1.6726766309170946) q[2];
rz(-0.7154658109276709) q[2];
ry(-2.9243149782645603) q[3];
rz(-0.2731640115263421) q[3];
ry(2.723028906675947) q[4];
rz(-1.2036195583761407) q[4];
ry(0.2907703513815919) q[5];
rz(-1.6745699168938817) q[5];
ry(-2.985283434285781) q[6];
rz(-1.214578045874232) q[6];
ry(-3.0852992510256345) q[7];
rz(-2.939874136438088) q[7];
ry(0.2732295798567132) q[8];
rz(-2.7237911170942897) q[8];
ry(-0.962990365608941) q[9];
rz(-2.783883989994706) q[9];
ry(-0.009377031831657945) q[10];
rz(3.1052577262824363) q[10];
ry(-1.8878509605771232) q[11];
rz(0.6202659156764687) q[11];
ry(2.20957871885318) q[12];
rz(-0.06666289989721373) q[12];
ry(1.6999352552044042) q[13];
rz(1.7297186003451452) q[13];
ry(0.204540894500191) q[14];
rz(0.06885883224210154) q[14];
ry(0.38438531692633937) q[15];
rz(-1.8245014025061925) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.5116460221892973) q[0];
rz(-0.1163398040191595) q[0];
ry(2.651734169492061) q[1];
rz(0.09332164947696986) q[1];
ry(3.1146001352796366) q[2];
rz(-0.03263225619902177) q[2];
ry(0.12929226646823064) q[3];
rz(2.246337050035452) q[3];
ry(-2.959162162423618) q[4];
rz(-1.6237824448174032) q[4];
ry(-0.5283704123491644) q[5];
rz(0.2641148083556226) q[5];
ry(3.045970846325318) q[6];
rz(1.1031961084543709) q[6];
ry(0.19481942388098344) q[7];
rz(-0.6895272775402601) q[7];
ry(-2.6964642172261106) q[8];
rz(2.328882532368562) q[8];
ry(2.2156067245463387) q[9];
rz(0.6909320607511181) q[9];
ry(0.6705690871336865) q[10];
rz(0.01892574297111081) q[10];
ry(-1.4032424755272013) q[11];
rz(0.8143691013470507) q[11];
ry(-0.000943331192114094) q[12];
rz(3.0329465452794384) q[12];
ry(-1.1412634650143028) q[13];
rz(2.716512488350538) q[13];
ry(-3.129249970291357) q[14];
rz(-2.3091904114221125) q[14];
ry(1.2348804308791843) q[15];
rz(0.49307939417724617) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.1599970372985915) q[0];
rz(-1.2088776440586295) q[0];
ry(-0.9083732446928094) q[1];
rz(-0.06737723867618772) q[1];
ry(3.0618680143203747) q[2];
rz(-2.6782834105642346) q[2];
ry(0.12473127617568523) q[3];
rz(1.0105977744771781) q[3];
ry(0.24006174205681763) q[4];
rz(2.070157213094501) q[4];
ry(-1.0378118728449819) q[5];
rz(-3.0233615918600862) q[5];
ry(-0.400468995908594) q[6];
rz(-1.9346398242245453) q[6];
ry(0.0029453132755294926) q[7];
rz(1.4142228920518305) q[7];
ry(-0.05433409688331096) q[8];
rz(0.8325830193583589) q[8];
ry(3.1242865508369793) q[9];
rz(0.8569533129319603) q[9];
ry(-0.005784881438013478) q[10];
rz(1.2019815203590307) q[10];
ry(-2.7806747055300804) q[11];
rz(-3.060253641624313) q[11];
ry(0.0008363524746056621) q[12];
rz(-2.2446131179423365) q[12];
ry(1.708260324118982) q[13];
rz(2.910127103093784) q[13];
ry(-0.0013162430803825842) q[14];
rz(-0.2137981452858325) q[14];
ry(-2.24365566597644) q[15];
rz(2.3794047917800762) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.44840169065945906) q[0];
rz(0.963161059705025) q[0];
ry(2.3554593659672247) q[1];
rz(0.3572555969430201) q[1];
ry(1.8752046624792962) q[2];
rz(1.3751946111991753) q[2];
ry(-3.0715578460177846) q[3];
rz(-2.0428758262400057) q[3];
ry(-1.7769947532417198) q[4];
rz(-0.4614266662867145) q[4];
ry(2.6095141161251356) q[5];
rz(-2.634012781286858) q[5];
ry(0.03543481164208018) q[6];
rz(-1.17498291924036) q[6];
ry(-3.126488579507304) q[7];
rz(2.1134727930155135) q[7];
ry(-0.4607187478211619) q[8];
rz(2.3524554169468184) q[8];
ry(0.6147997058553213) q[9];
rz(-1.4475629648560686) q[9];
ry(2.2726166594315127) q[10];
rz(2.2555628587582675) q[10];
ry(-0.7557002240151856) q[11];
rz(-1.8117726915963832) q[11];
ry(-3.140154638882846) q[12];
rz(-2.082966083909911) q[12];
ry(1.3504080618490075) q[13];
rz(1.6034436420876297) q[13];
ry(-3.1069735838330006) q[14];
rz(-2.955581168234499) q[14];
ry(1.4336878538944093) q[15];
rz(2.838440337876664) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.0338311258135227) q[0];
rz(-2.818406652828749) q[0];
ry(-1.314639774378103) q[1];
rz(-1.2251502979679278) q[1];
ry(3.1346735348605055) q[2];
rz(0.9668521681400453) q[2];
ry(-3.1393254319509434) q[3];
rz(1.2565386608880438) q[3];
ry(0.4792444409024945) q[4];
rz(2.4912841377257955) q[4];
ry(-0.799814170243071) q[5];
rz(-2.626899516345133) q[5];
ry(1.7355479544489167) q[6];
rz(2.5641769260999197) q[6];
ry(0.44486637039284727) q[7];
rz(-2.95629317396832) q[7];
ry(1.6626590738622469) q[8];
rz(3.0107319206332805) q[8];
ry(0.8175243360792315) q[9];
rz(-2.337420625051103) q[9];
ry(0.8725902316827012) q[10];
rz(2.7288246670800786) q[10];
ry(-3.114200920629667) q[11];
rz(-0.5512676601642825) q[11];
ry(-0.3154088551771048) q[12];
rz(-0.20207243282200418) q[12];
ry(-2.1428660904835843) q[13];
rz(1.7738643766625062) q[13];
ry(0.018111809639970736) q[14];
rz(2.561308133842549) q[14];
ry(2.1217232135662196) q[15];
rz(-2.7277275956311513) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.6227998331663365) q[0];
rz(-2.0956545493673344) q[0];
ry(2.163425828039445) q[1];
rz(-2.8029471536732595) q[1];
ry(-1.1912047908714645) q[2];
rz(-1.9843991386620674) q[2];
ry(-0.32871284063334344) q[3];
rz(1.3510865355916328) q[3];
ry(1.342732460012201) q[4];
rz(-0.3193328112884224) q[4];
ry(0.9249476598224744) q[5];
rz(3.0567997459357343) q[5];
ry(-0.002404813481724186) q[6];
rz(0.9298965268361563) q[6];
ry(0.011249617637109698) q[7];
rz(2.2908228961175268) q[7];
ry(-2.0443847731842233) q[8];
rz(-0.04268096435417698) q[8];
ry(-3.109469401460792) q[9];
rz(-0.19190057241818922) q[9];
ry(-1.3592300414861862) q[10];
rz(-0.9048813002234706) q[10];
ry(-3.125116209449289) q[11];
rz(-1.4313102445457224) q[11];
ry(-0.28567353800674133) q[12];
rz(-1.3838739235717172) q[12];
ry(2.418126689402179) q[13];
rz(-0.2850513726617301) q[13];
ry(-2.724187081069782) q[14];
rz(-0.875992871219867) q[14];
ry(1.9373494199532644) q[15];
rz(0.05036115424946152) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.4189662524842497) q[0];
rz(0.4668356049089955) q[0];
ry(2.973184793887053) q[1];
rz(-2.747662314634404) q[1];
ry(3.1227110587554274) q[2];
rz(1.932054780274966) q[2];
ry(3.138419751961148) q[3];
rz(0.5385646142920397) q[3];
ry(3.0285661047970627) q[4];
rz(-0.6042836783780322) q[4];
ry(1.4242548969330509) q[5];
rz(2.7941731980549025) q[5];
ry(-0.09102199032701819) q[6];
rz(-2.3577540411016256) q[6];
ry(0.025049901966316845) q[7];
rz(2.407151485010236) q[7];
ry(-1.3825452841441361) q[8];
rz(0.46069243324007675) q[8];
ry(3.128651079598318) q[9];
rz(-2.0518830738906706) q[9];
ry(1.0620651061183883) q[10];
rz(2.218782935514521) q[10];
ry(1.6442806946324335) q[11];
rz(2.723984802040259) q[11];
ry(2.9537587198495654) q[12];
rz(-2.680218551708381) q[12];
ry(-3.1310599364085547) q[13];
rz(2.6506030550940456) q[13];
ry(0.24094459623195763) q[14];
rz(-1.5334339544348778) q[14];
ry(2.8863111319613384) q[15];
rz(-0.2605853721732192) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.8114657942157022) q[0];
rz(2.3394902684371357) q[0];
ry(2.073214009274669) q[1];
rz(1.3700535053921632) q[1];
ry(-0.6521666018899612) q[2];
rz(-1.5321905531613513) q[2];
ry(0.017477253166513378) q[3];
rz(-2.567383512792927) q[3];
ry(2.6064043714768466) q[4];
rz(-2.5530053042616774) q[4];
ry(0.7303953869032329) q[5];
rz(0.3449866833836768) q[5];
ry(-3.1367289019898728) q[6];
rz(1.467317656681119) q[6];
ry(0.22602322586545934) q[7];
rz(-2.429917508056141) q[7];
ry(1.8947596891041867) q[8];
rz(1.5173353371627074) q[8];
ry(-3.02465438903179) q[9];
rz(0.7742659952372011) q[9];
ry(0.018424971901537468) q[10];
rz(-2.9202789092023824) q[10];
ry(-0.001519184228722992) q[11];
rz(1.2833661424663962) q[11];
ry(0.9851720434807084) q[12];
rz(1.5623366894647281) q[12];
ry(2.8350026281653204) q[13];
rz(0.6791633845737879) q[13];
ry(1.3196420250025003) q[14];
rz(-1.3880484312156967) q[14];
ry(-2.795366941954173) q[15];
rz(-1.067486778051321) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.497839060601543) q[0];
rz(3.031733529001653) q[0];
ry(-2.25657559225127) q[1];
rz(-2.4500893969987323) q[1];
ry(-1.4259977486461661) q[2];
rz(-3.12687626602785) q[2];
ry(0.007797490861177536) q[3];
rz(-2.5292468416472365) q[3];
ry(-2.766794819320512) q[4];
rz(-0.15862821078468417) q[4];
ry(-2.24764423996052) q[5];
rz(2.26448206867893) q[5];
ry(-2.628570317885395) q[6];
rz(1.7456505056201155) q[6];
ry(3.1200770821098733) q[7];
rz(0.7795822971596327) q[7];
ry(-0.043502650246737495) q[8];
rz(2.17942116330921) q[8];
ry(-3.1383522785750926) q[9];
rz(1.6767177110304585) q[9];
ry(-0.6919117347071388) q[10];
rz(-0.5381047777547608) q[10];
ry(0.36294198791494203) q[11];
rz(1.5217542805703483) q[11];
ry(-3.0488094183615204) q[12];
rz(-1.1020766974594185) q[12];
ry(3.140450629099182) q[13];
rz(0.27884274628498035) q[13];
ry(2.998779067765769) q[14];
rz(-0.35212687526500336) q[14];
ry(0.2901597761468744) q[15];
rz(0.07099758508326111) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.7947513728171138) q[0];
rz(-2.2016974905954987) q[0];
ry(3.120391714850786) q[1];
rz(2.377791735954816) q[1];
ry(2.588730844234537) q[2];
rz(2.976226574640268) q[2];
ry(1.234455689540047) q[3];
rz(-2.601438011990622) q[3];
ry(2.2037054208063642) q[4];
rz(2.3424145629548554) q[4];
ry(-0.03902284312806) q[5];
rz(2.3856298803750606) q[5];
ry(-0.06472398119265589) q[6];
rz(-1.6944380054123416) q[6];
ry(2.6360953552938717) q[7];
rz(1.7782138175331736) q[7];
ry(0.9096053509994771) q[8];
rz(-2.6182768637234264) q[8];
ry(2.9020429103197056) q[9];
rz(-0.17784897010211145) q[9];
ry(-3.070573824086961) q[10];
rz(0.6128538258712029) q[10];
ry(0.009204870036183976) q[11];
rz(-2.886259925817609) q[11];
ry(1.948616000487997) q[12];
rz(-1.7465948356152525) q[12];
ry(-2.198482519313128) q[13];
rz(1.624668885052655) q[13];
ry(1.16613033076984) q[14];
rz(1.0998639987666496) q[14];
ry(-2.881576203512227) q[15];
rz(2.4698876442079842) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.669891784189349) q[0];
rz(-1.8180465019482694) q[0];
ry(-1.312595132708439) q[1];
rz(-0.32276813765909557) q[1];
ry(2.4599019180357873) q[2];
rz(0.033690929211510484) q[2];
ry(-2.8392877095671074) q[3];
rz(-2.4711256556551007) q[3];
ry(-3.1359068434307034) q[4];
rz(0.4845762853697204) q[4];
ry(-1.2334000662104454) q[5];
rz(1.0750505140156374) q[5];
ry(-0.4217886747552866) q[6];
rz(2.318023348287798) q[6];
ry(2.700934350799046) q[7];
rz(2.0923027541717047) q[7];
ry(3.131884446717677) q[8];
rz(0.19610231740482667) q[8];
ry(-1.0079442601580049) q[9];
rz(1.377214750751345) q[9];
ry(2.6593424204243736) q[10];
rz(1.3892676146840002) q[10];
ry(-2.3266156244276623) q[11];
rz(0.5061525902504493) q[11];
ry(-3.057747926401266) q[12];
rz(-0.7841848872673459) q[12];
ry(-0.8709779117785317) q[13];
rz(-1.1431501597735332) q[13];
ry(2.3559661513466255) q[14];
rz(1.4110785252470839) q[14];
ry(-2.572242609502025) q[15];
rz(-0.9653434018952715) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.3510220968702307) q[0];
rz(2.566197872395281) q[0];
ry(-0.8696703197117933) q[1];
rz(3.1344827352266544) q[1];
ry(-0.4421256496375569) q[2];
rz(-2.875589118137541) q[2];
ry(1.955157163184903) q[3];
rz(-1.2742133022581505) q[3];
ry(-0.11999555917524507) q[4];
rz(0.09990826367873495) q[4];
ry(-0.08393115136049935) q[5];
rz(-1.0874458627225205) q[5];
ry(-3.1027674887238192) q[6];
rz(2.218637595723626) q[6];
ry(2.7381811932381246) q[7];
rz(-1.581898827290554) q[7];
ry(0.013773015455605265) q[8];
rz(0.19812305020644952) q[8];
ry(0.2167685565902817) q[9];
rz(1.9757047067720261) q[9];
ry(0.6404711629895723) q[10];
rz(2.638512459299759) q[10];
ry(3.076941243075531) q[11];
rz(2.118829245630879) q[11];
ry(-0.05610985832004626) q[12];
rz(1.917218724216439) q[12];
ry(-3.017204843770047) q[13];
rz(2.2952180958882216) q[13];
ry(-0.6202288287897062) q[14];
rz(0.5903462389110911) q[14];
ry(2.9345298901809382) q[15];
rz(-0.2691217234970898) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.3936669781533755) q[0];
rz(0.3356855593558145) q[0];
ry(-0.07654082760134066) q[1];
rz(-2.9914286954199105) q[1];
ry(-3.1412003677429605) q[2];
rz(-2.893212449806301) q[2];
ry(-0.01436638283551428) q[3];
rz(-1.5636833040958789) q[3];
ry(2.29102668549619) q[4];
rz(1.914987935571908) q[4];
ry(2.257805764485499) q[5];
rz(0.05538363190865739) q[5];
ry(0.16774551493637027) q[6];
rz(2.2364152976690517) q[6];
ry(2.2171418632205424) q[7];
rz(1.7099630537513228) q[7];
ry(-3.13227944490232) q[8];
rz(-0.27419410742244477) q[8];
ry(0.027816194497218838) q[9];
rz(1.8237308875553433) q[9];
ry(-3.076270662013183) q[10];
rz(-0.3463416964466531) q[10];
ry(0.08763505895968102) q[11];
rz(0.8206117121192688) q[11];
ry(0.042635849680453614) q[12];
rz(-1.6944525653279192) q[12];
ry(-1.8804159026247456) q[13];
rz(0.21561326346539275) q[13];
ry(1.8429749404353553) q[14];
rz(0.12368182307487084) q[14];
ry(1.3739252428764095) q[15];
rz(0.5228547592646134) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.015554309871778) q[0];
rz(-2.646608470872766) q[0];
ry(-0.848082609366644) q[1];
rz(0.2602972213623662) q[1];
ry(0.3167411423220363) q[2];
rz(2.9569912404019787) q[2];
ry(-3.1048886327945575) q[3];
rz(-1.16906983690569) q[3];
ry(0.5052670349781777) q[4];
rz(-3.066654549031969) q[4];
ry(-0.02259966253799206) q[5];
rz(2.3159324740471217) q[5];
ry(3.088057177047227) q[6];
rz(-0.7176400416054333) q[6];
ry(0.19513394277728316) q[7];
rz(1.2460675792995297) q[7];
ry(-3.135094850993542) q[8];
rz(-0.24126416129910777) q[8];
ry(0.3710402788045864) q[9];
rz(0.059186417358235474) q[9];
ry(0.6432766984550611) q[10];
rz(0.8628165738111839) q[10];
ry(-3.0733110557911765) q[11];
rz(-0.9053260199927378) q[11];
ry(0.019308503609012817) q[12];
rz(-2.659707637576898) q[12];
ry(-0.2652645977262447) q[13];
rz(2.484801913439444) q[13];
ry(-0.04532021828294042) q[14];
rz(1.7862093238196333) q[14];
ry(-2.9326611812277434) q[15];
rz(-1.7172919941097575) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.0400588534609234) q[0];
rz(0.7713435197925405) q[0];
ry(1.3118048218891218) q[1];
rz(-2.843664168600297) q[1];
ry(-3.126938837155084) q[2];
rz(-0.3630407934244175) q[2];
ry(1.44747364517226) q[3];
rz(2.9006421444443755) q[3];
ry(-2.147442219920599) q[4];
rz(-0.3831788928331378) q[4];
ry(-0.04544204343195703) q[5];
rz(-0.733005518278218) q[5];
ry(-1.291717025079354) q[6];
rz(2.3884005360983083) q[6];
ry(-0.9088193312066287) q[7];
rz(-2.1660908164972614) q[7];
ry(2.990747601968629) q[8];
rz(0.018993494623702922) q[8];
ry(-1.9310923257159207) q[9];
rz(-2.9949099850917675) q[9];
ry(1.9204361384233286) q[10];
rz(0.05248136239690445) q[10];
ry(0.09778331767863112) q[11];
rz(2.4463883293275033) q[11];
ry(2.6882195487221856) q[12];
rz(-0.2006617806575596) q[12];
ry(-2.3838475874334226) q[13];
rz(-0.9295372580604866) q[13];
ry(-2.5989468219503298) q[14];
rz(-2.351908393369659) q[14];
ry(-0.3306167957321078) q[15];
rz(2.2039801083524977) q[15];