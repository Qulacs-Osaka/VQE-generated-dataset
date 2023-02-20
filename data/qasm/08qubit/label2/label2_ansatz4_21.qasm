OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.5675535742933067) q[0];
rz(-1.7153021442484668) q[0];
ry(-0.6622926797397313) q[1];
rz(-1.91946116912281) q[1];
ry(-3.1411730506558913) q[2];
rz(2.2911239744702447) q[2];
ry(1.2253195849450963e-05) q[3];
rz(2.592971978604059) q[3];
ry(1.5723620630010293) q[4];
rz(3.084790032048379) q[4];
ry(-1.6074558246764163) q[5];
rz(2.099315267370889) q[5];
ry(3.1398240289047643) q[6];
rz(-3.1366933740374954) q[6];
ry(-3.1398614515478225) q[7];
rz(-2.7023600454287395) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.4050394837952416) q[0];
rz(1.7975789468574055) q[0];
ry(-1.5561476032199166) q[1];
rz(0.021243312668469905) q[1];
ry(1.664479113116345) q[2];
rz(1.5766759420240959) q[2];
ry(0.14831013784157931) q[3];
rz(-1.8845441101523033) q[3];
ry(1.6055151854292318) q[4];
rz(2.3308949562753187) q[4];
ry(-3.075526088219231) q[5];
rz(-2.2553398049955353) q[5];
ry(-1.3786712655322626) q[6];
rz(-0.6844591014567811) q[6];
ry(1.4503851543217072) q[7];
rz(-1.76772994276621) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.0028848906371377) q[0];
rz(1.944431139801082) q[0];
ry(-1.6527349726242755) q[1];
rz(2.3956397538841223) q[1];
ry(-0.8796580412040198) q[2];
rz(-1.13407690099105) q[2];
ry(-3.026148299306712) q[3];
rz(1.3229024449526356) q[3];
ry(-3.130158875158193) q[4];
rz(0.9200445428575463) q[4];
ry(0.003732576422573075) q[5];
rz(-0.15607743960855777) q[5];
ry(-2.593952389806122) q[6];
rz(2.6698617818947925) q[6];
ry(2.3745258452763673) q[7];
rz(-0.5779606105546282) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.4701770172070547) q[0];
rz(-0.441382221829004) q[0];
ry(1.593022025235418) q[1];
rz(-1.802770789375037) q[1];
ry(0.30835018187417196) q[2];
rz(1.680756637409086) q[2];
ry(-1.6315865925743929) q[3];
rz(0.8687274364517474) q[3];
ry(1.525490044444803) q[4];
rz(-1.7990260924567816) q[4];
ry(1.6107242027232216) q[5];
rz(-1.509073774242832) q[5];
ry(2.777512914148021) q[6];
rz(1.163003392581536) q[6];
ry(-1.9907875132945412) q[7];
rz(1.5477765249591215) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.0157360045672292) q[0];
rz(-2.7325586920488267) q[0];
ry(1.5045544691365593) q[1];
rz(2.57032881543973) q[1];
ry(3.1295819543268992) q[2];
rz(2.7826010943454644) q[2];
ry(-3.1278904127222287) q[3];
rz(-0.6017303688015688) q[3];
ry(3.053670293183993) q[4];
rz(-0.6111976104225212) q[4];
ry(-3.0121196774895362) q[5];
rz(0.3251368718391067) q[5];
ry(2.749331804643658) q[6];
rz(-2.6220279799227555) q[6];
ry(1.7040726545577634) q[7];
rz(-2.1300773954062846) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.931515935380931) q[0];
rz(-2.94250731486389) q[0];
ry(0.6313654700097313) q[1];
rz(0.718875559064817) q[1];
ry(-2.6910409637030996) q[2];
rz(2.188521176780185) q[2];
ry(2.690687889153996) q[3];
rz(1.6554196345189234) q[3];
ry(-2.2510425454314644) q[4];
rz(-1.5132330917117764) q[4];
ry(-2.243915189343693) q[5];
rz(-2.124788488067081) q[5];
ry(-2.4127705141849085) q[6];
rz(-0.756199108873768) q[6];
ry(1.7581648137578298) q[7];
rz(-2.0651791206954426) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.7548767091366155) q[0];
rz(1.181801373387562) q[0];
ry(-2.589060685190813) q[1];
rz(-1.3874120180436318) q[1];
ry(-3.080924315454749) q[2];
rz(0.5991725240977526) q[2];
ry(-0.06644919774337321) q[3];
rz(-0.18549022556495132) q[3];
ry(3.08440054321134) q[4];
rz(2.8182167432792293) q[4];
ry(3.0952463990178374) q[5];
rz(-0.05989787827765895) q[5];
ry(2.999727997767416) q[6];
rz(2.9647262650611674) q[6];
ry(-0.6150771209450411) q[7];
rz(3.0152860171256286) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.6531472229957147) q[0];
rz(-1.8314779562597978) q[0];
ry(0.7088140468861805) q[1];
rz(-1.5413891425006359) q[1];
ry(-1.6808649521472228) q[2];
rz(-0.8429168230942876) q[2];
ry(2.3829791016032913) q[3];
rz(2.641248914852119) q[3];
ry(2.0088419316546107) q[4];
rz(-1.7251178357534325) q[4];
ry(-1.2071853231160157) q[5];
rz(-1.570462720721903) q[5];
ry(-2.34740504452925) q[6];
rz(-1.7138639083088325) q[6];
ry(-0.4924452857747742) q[7];
rz(0.7790842375872296) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.928583011201413) q[0];
rz(-2.881239817161428) q[0];
ry(-2.278826716769944) q[1];
rz(0.2410144600845991) q[1];
ry(-0.6150163776411162) q[2];
rz(2.5087234677068246) q[2];
ry(-0.8608992962257834) q[3];
rz(-1.0597101196345635) q[3];
ry(-1.9674172738171398) q[4];
rz(-1.1046667123238025) q[4];
ry(-1.967260041371958) q[5];
rz(2.0515732051820437) q[5];
ry(3.0174757239730328) q[6];
rz(-3.07070237231543) q[6];
ry(2.657449211334296) q[7];
rz(-1.2341711559880497) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.1638314123572906) q[0];
rz(-2.35053305585037) q[0];
ry(1.1451253107553592) q[1];
rz(1.3969393769929825) q[1];
ry(-1.8359795673937813) q[2];
rz(-2.847811358664503) q[2];
ry(1.6620512927179103) q[3];
rz(0.1025670460801762) q[3];
ry(-2.0789612519333875) q[4];
rz(-0.9244241541249141) q[4];
ry(2.010494419693105) q[5];
rz(-0.8918560530721855) q[5];
ry(-3.0938559887860184) q[6];
rz(-0.21184408970586954) q[6];
ry(-1.2917689450605234) q[7];
rz(-0.8769845199694873) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.7912691533358646) q[0];
rz(-1.4595736118023053) q[0];
ry(0.6929323392242042) q[1];
rz(1.2652970648551072) q[1];
ry(-2.562274825821015) q[2];
rz(0.31682735953793745) q[2];
ry(1.7233405407508418) q[3];
rz(-0.5447842240520311) q[3];
ry(-1.1769583487597595) q[4];
rz(-0.18539906968754535) q[4];
ry(0.8678376024950443) q[5];
rz(-0.26155584757149697) q[5];
ry(-2.7462761003384712) q[6];
rz(3.0223524590480433) q[6];
ry(-0.11396188019881581) q[7];
rz(-2.113304996061573) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.6395398330712494) q[0];
rz(1.3363874246920506) q[0];
ry(0.28762831303317515) q[1];
rz(-1.8499646750925018) q[1];
ry(3.1327756467538412) q[2];
rz(1.9273402406452425) q[2];
ry(0.0028441246235164637) q[3];
rz(-0.9292912861595166) q[3];
ry(-1.5776285852030796) q[4];
rz(2.941724585706378) q[4];
ry(1.5794334380952249) q[5];
rz(-0.1962677199730196) q[5];
ry(3.0322933571912145) q[6];
rz(0.0466464478617092) q[6];
ry(1.3938296248848234) q[7];
rz(-0.9343583010034954) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.9441293124476786) q[0];
rz(2.5104307285091045) q[0];
ry(2.6319578113951407) q[1];
rz(0.04408144720879293) q[1];
ry(-1.2933882865742523) q[2];
rz(-0.005348793419197584) q[2];
ry(-1.9468689479280532) q[3];
rz(-2.1123959261552274) q[3];
ry(0.9436929884014447) q[4];
rz(0.5934463316527433) q[4];
ry(-2.271725213492862) q[5];
rz(-1.5562904689617099) q[5];
ry(-2.759576264936957) q[6];
rz(-1.546671355726076) q[6];
ry(1.0444824746653771) q[7];
rz(2.4519686575524564) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.8462481436038392) q[0];
rz(-0.993868337530242) q[0];
ry(2.282368756697676) q[1];
rz(-3.0258659096011424) q[1];
ry(-0.0008759682318020268) q[2];
rz(-2.7928470388045543) q[2];
ry(-0.164759707906403) q[3];
rz(2.3403594245159853) q[3];
ry(-3.084588351134198) q[4];
rz(-0.8423065044515635) q[4];
ry(-0.0014814159247920054) q[5];
rz(-2.4010550708238667) q[5];
ry(0.0584469467446489) q[6];
rz(-0.07369330689547748) q[6];
ry(-0.10652076820768248) q[7];
rz(1.3574396011424668) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5786934974436504) q[0];
rz(0.48677030755813266) q[0];
ry(-0.01262735600533027) q[1];
rz(1.6608456490553052) q[1];
ry(0.018087205574313446) q[2];
rz(-1.3826297259251188) q[2];
ry(0.013769008192777044) q[3];
rz(-1.3989554525619854) q[3];
ry(-0.1440508266773746) q[4];
rz(2.70699222264779) q[4];
ry(-0.1693247267514284) q[5];
rz(-1.2823568216365056) q[5];
ry(0.2968778117490274) q[6];
rz(-2.643035065363571) q[6];
ry(2.629508803640013) q[7];
rz(-0.12339667898098572) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.8387225440546047) q[0];
rz(1.7486681471568186) q[0];
ry(-2.07397745525153) q[1];
rz(-3.1361053434287243) q[1];
ry(-1.7298613012769248) q[2];
rz(-0.6954333408475604) q[2];
ry(1.4884327949719875) q[3];
rz(1.0607857949773996) q[3];
ry(1.5593918898827734) q[4];
rz(2.4504405721267792) q[4];
ry(-1.555043865089559) q[5];
rz(3.135473456242176) q[5];
ry(-1.068014334632824) q[6];
rz(-1.6092525001756266) q[6];
ry(-0.4367693618461086) q[7];
rz(1.2520806825053183) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.663750575670106) q[0];
rz(1.6150379968736603) q[0];
ry(1.0483923323797235) q[1];
rz(2.7722097716883716) q[1];
ry(0.315286190142707) q[2];
rz(-0.5433923757025845) q[2];
ry(2.9015167810969205) q[3];
rz(0.003038587769617986) q[3];
ry(-3.0797508564361595) q[4];
rz(-2.6834248478609646) q[4];
ry(-3.1389868762024666) q[5];
rz(-1.7028278752475376) q[5];
ry(-0.6119517995500435) q[6];
rz(2.2545176381047005) q[6];
ry(-0.6724850932783011) q[7];
rz(-2.405359745935241) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.0443845057676366) q[0];
rz(-0.8340321868431751) q[0];
ry(-0.00691338567806153) q[1];
rz(-1.6339012961915609) q[1];
ry(3.109364242957933) q[2];
rz(-2.502774799375425) q[2];
ry(-3.120478316638459) q[3];
rz(1.1953220007989076) q[3];
ry(-0.5996702153530161) q[4];
rz(2.101186356933459) q[4];
ry(1.021162777274009) q[5];
rz(2.0004718593853523) q[5];
ry(0.16639224833242242) q[6];
rz(-2.932598485971949) q[6];
ry(-0.12800368519318006) q[7];
rz(2.183636119243695) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.8958937490485137) q[0];
rz(-2.800113355273015) q[0];
ry(-0.9227232469944558) q[1];
rz(2.235548795984554) q[1];
ry(-0.9526649264566426) q[2];
rz(-0.28958605195211806) q[2];
ry(-0.9528866246308656) q[3];
rz(-0.43482676935493636) q[3];
ry(-1.8992786378310953) q[4];
rz(-2.20298125944296) q[4];
ry(-1.2871637438480306) q[5];
rz(-0.5674003639845299) q[5];
ry(0.4738772055960325) q[6];
rz(-1.3764913218092947) q[6];
ry(-2.692264386068804) q[7];
rz(2.7775069988886605) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.502251266269558) q[0];
rz(-0.9226864008002379) q[0];
ry(-0.7109434148229159) q[1];
rz(0.8099777170509429) q[1];
ry(-1.8171953600269681) q[2];
rz(3.0240990199362723) q[2];
ry(-1.7019211538176238) q[3];
rz(-0.3332676184632497) q[3];
ry(0.07574043890150595) q[4];
rz(2.83250261766037) q[4];
ry(3.0702941192944055) q[5];
rz(0.16679964727689892) q[5];
ry(-1.8268163046003627) q[6];
rz(-0.9252572665968966) q[6];
ry(2.8369973564879443) q[7];
rz(0.510246555430589) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.8262422388986157) q[0];
rz(0.7633399706080182) q[0];
ry(-0.8586237679681296) q[1];
rz(-1.4439552520454457) q[1];
ry(1.3715183656527543) q[2];
rz(-0.7444972610661926) q[2];
ry(1.7769412400968383) q[3];
rz(0.3525135796376711) q[3];
ry(-2.1034319518828597) q[4];
rz(0.9449319493604349) q[4];
ry(3.056306823074951) q[5];
rz(-1.1298005887811189) q[5];
ry(-1.2661361990510107) q[6];
rz(-0.6060380689332765) q[6];
ry(-1.167787805509841) q[7];
rz(-1.90262061835442) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.5400391397679076) q[0];
rz(-2.9155254835551503) q[0];
ry(0.9824987408855694) q[1];
rz(2.907226025452782) q[1];
ry(0.07372022175061232) q[2];
rz(-3.113460633117794) q[2];
ry(0.04717182108655746) q[3];
rz(2.272102855789948) q[3];
ry(2.545855304713161) q[4];
rz(-2.39935588528316) q[4];
ry(2.4259412146678527) q[5];
rz(1.7849105429492393) q[5];
ry(-3.107459850212889) q[6];
rz(2.837263455909763) q[6];
ry(-3.133143884182741) q[7];
rz(1.3296060241827474) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.4870525275106679) q[0];
rz(0.349392055562455) q[0];
ry(0.855267692313806) q[1];
rz(-0.047756708563621894) q[1];
ry(-3.138484714543479) q[2];
rz(1.5054041244270193) q[2];
ry(-3.131849719354621) q[3];
rz(0.8199549106048619) q[3];
ry(-2.8780511639110813) q[4];
rz(1.3165060730888032) q[4];
ry(-0.3675046924126383) q[5];
rz(1.523955262974587) q[5];
ry(-1.6604210765710803) q[6];
rz(0.10922117216015881) q[6];
ry(-1.7351426876118756) q[7];
rz(-1.1465996373909837) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.4079396810132998) q[0];
rz(1.8551181795997502) q[0];
ry(0.16559250067545075) q[1];
rz(-0.2289518249984853) q[1];
ry(0.9253900941203639) q[2];
rz(0.34895686076636157) q[2];
ry(2.2738565284026087) q[3];
rz(2.911755949271408) q[3];
ry(-1.5236879846896025) q[4];
rz(2.3323908877148445) q[4];
ry(1.9737859420747248) q[5];
rz(-1.5465291337811822) q[5];
ry(0.001087982198172632) q[6];
rz(-1.138287198834484) q[6];
ry(3.1393228446701755) q[7];
rz(3.0581786593477176) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.117695381841971) q[0];
rz(0.8845921032463772) q[0];
ry(1.1688075544763388) q[1];
rz(0.8212307850363328) q[1];
ry(2.6663710739198003) q[2];
rz(-2.0203290491918104) q[2];
ry(2.931472126690681) q[3];
rz(-2.571114450467579) q[3];
ry(0.07061979057044701) q[4];
rz(0.7552823520991732) q[4];
ry(-2.219096410172924) q[5];
rz(-2.132488432063358) q[5];
ry(-1.269937600281902) q[6];
rz(-2.384803659990257) q[6];
ry(1.9414551029119007) q[7];
rz(-2.3148639146830496) q[7];