OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-3.0046390864465318) q[0];
rz(-0.8569493441786289) q[0];
ry(-2.1118769049037223) q[1];
rz(-3.0890087735437857) q[1];
ry(0.009798056077604556) q[2];
rz(0.9126242363453543) q[2];
ry(-1.8375764040547515) q[3];
rz(2.7065390456630776) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.897394124907958) q[0];
rz(2.9286009755618743) q[0];
ry(2.335364069793896) q[1];
rz(-2.2112054592020525) q[1];
ry(-1.460914116810305) q[2];
rz(-0.5339869219665276) q[2];
ry(-1.124552708869615) q[3];
rz(-0.39321783003605676) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.0951797219147323) q[0];
rz(2.3315769593033275) q[0];
ry(-1.3712567902114567) q[1];
rz(-1.9065327542032833) q[1];
ry(-0.6675848803128507) q[2];
rz(-0.6343206687399001) q[2];
ry(1.756997225059945) q[3];
rz(-2.546851883041804) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.767084409315448) q[0];
rz(-1.3944670795192335) q[0];
ry(2.8473796688203112) q[1];
rz(-1.4968122072499142) q[1];
ry(-1.7991157303953822) q[2];
rz(-2.200864171920837) q[2];
ry(1.4418346253492953) q[3];
rz(-2.3079986324106843) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.62534076951276) q[0];
rz(-0.23361827480430983) q[0];
ry(-0.6502066128166994) q[1];
rz(-1.555782941736905) q[1];
ry(-2.6411673315610433) q[2];
rz(0.6003284281476616) q[2];
ry(0.4418458693780343) q[3];
rz(0.6777366638557974) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.3101291885945497) q[0];
rz(-1.1081373777349224) q[0];
ry(2.079863365437319) q[1];
rz(2.6972711364417497) q[1];
ry(-1.875714106701588) q[2];
rz(3.1228941856295265) q[2];
ry(-2.9030784874474027) q[3];
rz(-0.2549073059046559) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.267809055997925) q[0];
rz(2.049677887675628) q[0];
ry(0.7017682393026492) q[1];
rz(-2.3321990833438266) q[1];
ry(-1.7640571080761784) q[2];
rz(3.0039146771786696) q[2];
ry(-1.2804498001367417) q[3];
rz(-2.873019128987522) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.7145650836742972) q[0];
rz(0.38258807144276563) q[0];
ry(-2.8385810894385815) q[1];
rz(0.7011720659860834) q[1];
ry(-1.8765046115230217) q[2];
rz(0.9392771998244392) q[2];
ry(-2.0521808078943646) q[3];
rz(-1.800451576017645) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.081916164294176) q[0];
rz(-0.8820122189053548) q[0];
ry(-0.5248766523338467) q[1];
rz(-1.262557022995314) q[1];
ry(2.2022226404826624) q[2];
rz(2.8476617238950137) q[2];
ry(-2.906224514112609) q[3];
rz(-1.623905306157672) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.244996581697959) q[0];
rz(-0.2120687869448581) q[0];
ry(-2.440698103215754) q[1];
rz(-1.394803190060669) q[1];
ry(1.5607016186709526) q[2];
rz(-1.2103744420799618) q[2];
ry(0.34377413144371083) q[3];
rz(1.7686208795578935) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9653731172015628) q[0];
rz(2.1290619070179906) q[0];
ry(1.3248151763013267) q[1];
rz(1.3042127915351394) q[1];
ry(2.3759087620238075) q[2];
rz(-1.669022852137642) q[2];
ry(-2.6276252065842725) q[3];
rz(2.4415264075277756) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.3034096874619476) q[0];
rz(-0.5731384401144313) q[0];
ry(0.22591474835456446) q[1];
rz(-3.033556626641538) q[1];
ry(1.9011965992448085) q[2];
rz(-2.9730645447858115) q[2];
ry(0.5437259254138507) q[3];
rz(0.048509043758959154) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0707219983696157) q[0];
rz(-1.1123886229298054) q[0];
ry(2.416964135047877) q[1];
rz(1.8741035386891554) q[1];
ry(2.626788016353307) q[2];
rz(0.3958696507274721) q[2];
ry(-2.917129494892468) q[3];
rz(-2.5659737485009506) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6451595040110116) q[0];
rz(0.4179843148095159) q[0];
ry(0.515527464402437) q[1];
rz(0.7437840168011154) q[1];
ry(-2.0582920048539193) q[2];
rz(0.8251711980046696) q[2];
ry(-1.8649765120265842) q[3];
rz(0.3537869877770883) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.012750876670518) q[0];
rz(1.0224794239531985) q[0];
ry(0.04699481636058944) q[1];
rz(1.9537810513407363) q[1];
ry(-2.3142169642798502) q[2];
rz(0.12910007899358267) q[2];
ry(-0.33816106296761106) q[3];
rz(-2.9772736128807744) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.4806982057980997) q[0];
rz(-2.747550266988937) q[0];
ry(1.327673996566569) q[1];
rz(0.2910652634574192) q[1];
ry(-1.7625054166986875) q[2];
rz(0.610187582524202) q[2];
ry(-0.9077187667879363) q[3];
rz(2.061430145732106) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.7693903804075677) q[0];
rz(0.5785705165985703) q[0];
ry(-1.9628242767329953) q[1];
rz(1.4767744443378987) q[1];
ry(0.683223581834671) q[2];
rz(0.2372402702439125) q[2];
ry(1.1866487249528088) q[3];
rz(-0.326898231684516) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.811709976635674) q[0];
rz(0.008399987423598711) q[0];
ry(2.5697053086976185) q[1];
rz(0.8544600905368872) q[1];
ry(-0.2719813844122632) q[2];
rz(2.461272763381892) q[2];
ry(2.200226134315986) q[3];
rz(-1.1708756280346773) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6384372434850012) q[0];
rz(0.22787251134247646) q[0];
ry(-2.7318528943131257) q[1];
rz(-0.5773065856880893) q[1];
ry(-1.4437328706286792) q[2];
rz(2.427888828476347) q[2];
ry(-0.42489380631830453) q[3];
rz(-2.409349541853965) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.9135554166334545) q[0];
rz(-1.1515192160231988) q[0];
ry(-0.5507169936063585) q[1];
rz(0.05998862588178517) q[1];
ry(1.1927827124543364) q[2];
rz(1.3844899161182613) q[2];
ry(0.2510334513643946) q[3];
rz(-0.032544206143426894) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.5781467295192708) q[0];
rz(2.3512976449088665) q[0];
ry(-2.976341726860141) q[1];
rz(-2.5209557027532576) q[1];
ry(-2.9484586940811783) q[2];
rz(0.7376578614639421) q[2];
ry(-2.735325974027697) q[3];
rz(1.6102409573807916) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.530345754152968) q[0];
rz(2.4757479986936537) q[0];
ry(1.5586616930012966) q[1];
rz(-1.9084583175721674) q[1];
ry(0.15169375120858217) q[2];
rz(0.035090908228855966) q[2];
ry(0.28314219607418506) q[3];
rz(0.6712218515901233) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.33690878755303777) q[0];
rz(2.918958813484099) q[0];
ry(-1.5493612780010437) q[1];
rz(-0.3747408238617558) q[1];
ry(-0.4764267209166626) q[2];
rz(2.299734654218279) q[2];
ry(1.3955928721854987) q[3];
rz(-1.7592523528323492) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.363809729995643) q[0];
rz(-2.184656305971839) q[0];
ry(2.4297572802481837) q[1];
rz(2.3463617458227892) q[1];
ry(0.3699285116493404) q[2];
rz(1.0272421307975441) q[2];
ry(0.9718678256373003) q[3];
rz(-2.725354046135884) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.491695992603684) q[0];
rz(1.77122784710732) q[0];
ry(0.8367807198500979) q[1];
rz(-2.862632455879008) q[1];
ry(-3.140205829946522) q[2];
rz(-3.0678794668593325) q[2];
ry(-2.335094696456732) q[3];
rz(0.7316688750330824) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.5507367973250146) q[0];
rz(-2.060276353824538) q[0];
ry(2.955277386625763) q[1];
rz(-3.0931259028260074) q[1];
ry(-0.3740018118174646) q[2];
rz(-1.231116313161839) q[2];
ry(-1.9004656730979468) q[3];
rz(-0.8056650754826578) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6158325926195287) q[0];
rz(-3.1217261811210646) q[0];
ry(-2.495701623538685) q[1];
rz(-0.3353892433708179) q[1];
ry(-1.8059736787624012) q[2];
rz(-0.7697282451715062) q[2];
ry(0.8730149180690275) q[3];
rz(0.3777582398236161) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.0575674971276414) q[0];
rz(1.2554442667194925) q[0];
ry(0.8153532294358197) q[1];
rz(-1.0908585511083397) q[1];
ry(1.4460165896754011) q[2];
rz(2.731000432772291) q[2];
ry(-0.23144332122733502) q[3];
rz(-1.594935104738346) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.7760137571752646) q[0];
rz(-0.18624340281426388) q[0];
ry(0.8157697968943278) q[1];
rz(2.3203211368926966) q[1];
ry(-1.6600465924656589) q[2];
rz(2.592536376706828) q[2];
ry(-2.3156133304563213) q[3];
rz(1.5904696515477568) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.36154662924905356) q[0];
rz(1.5486776395483002) q[0];
ry(-2.3077761621591786) q[1];
rz(0.6546558972422949) q[1];
ry(0.9737821761137926) q[2];
rz(0.1689231238879727) q[2];
ry(1.9091301888240224) q[3];
rz(0.13382208382976296) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.1753597935008706) q[0];
rz(0.9821887402977443) q[0];
ry(0.5760773349423086) q[1];
rz(2.887171682869182) q[1];
ry(-0.7735881818057765) q[2];
rz(-2.3454323288817918) q[2];
ry(-3.081862461891486) q[3];
rz(-2.2104175875396854) q[3];