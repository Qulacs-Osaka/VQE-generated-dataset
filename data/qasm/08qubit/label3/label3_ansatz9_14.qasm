OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.1939597019077715) q[0];
ry(-2.966030516555665) q[1];
cx q[0],q[1];
ry(-2.906734730806636) q[0];
ry(0.025209962332637528) q[1];
cx q[0],q[1];
ry(2.1987829474467104) q[2];
ry(1.4870857763764613) q[3];
cx q[2],q[3];
ry(-1.9366134487593065) q[2];
ry(1.4067060296366545) q[3];
cx q[2],q[3];
ry(-0.8875153642673704) q[4];
ry(2.5128588497526327) q[5];
cx q[4],q[5];
ry(0.6449745998352098) q[4];
ry(2.4022468869462834) q[5];
cx q[4],q[5];
ry(1.119842608399697) q[6];
ry(0.8565938509422235) q[7];
cx q[6],q[7];
ry(-2.299262808365972) q[6];
ry(2.489611763746624) q[7];
cx q[6],q[7];
ry(-0.9066609107081239) q[0];
ry(-2.356530134115299) q[2];
cx q[0],q[2];
ry(-0.6512212227290758) q[0];
ry(-2.3824238601905687) q[2];
cx q[0],q[2];
ry(-0.9021552686566681) q[2];
ry(-1.576748167496036) q[4];
cx q[2],q[4];
ry(1.5239446333626099) q[2];
ry(-0.44923554466072346) q[4];
cx q[2],q[4];
ry(-1.1738509203041418) q[4];
ry(2.1072718223488764) q[6];
cx q[4],q[6];
ry(2.303943233366168) q[4];
ry(-0.6637713379117539) q[6];
cx q[4],q[6];
ry(-0.3565486884878925) q[1];
ry(0.7312465038335683) q[3];
cx q[1],q[3];
ry(2.68135133434067) q[1];
ry(2.060093902499273) q[3];
cx q[1],q[3];
ry(-2.8085403422635413) q[3];
ry(1.4862218591455028) q[5];
cx q[3],q[5];
ry(-0.6185159025025189) q[3];
ry(1.307012594955293) q[5];
cx q[3],q[5];
ry(-2.4840985559503936) q[5];
ry(-0.4940730059835589) q[7];
cx q[5],q[7];
ry(-0.5804100557792928) q[5];
ry(1.0025151117846018) q[7];
cx q[5],q[7];
ry(-0.7020532605295687) q[0];
ry(-1.8152017760360968) q[3];
cx q[0],q[3];
ry(-1.3897473635949091) q[0];
ry(0.23004333775344551) q[3];
cx q[0],q[3];
ry(1.5381404636336717) q[1];
ry(1.0644255090857753) q[2];
cx q[1],q[2];
ry(-1.823260985110306) q[1];
ry(0.8126592006773107) q[2];
cx q[1],q[2];
ry(0.6397250521063674) q[2];
ry(0.5494935439893824) q[5];
cx q[2],q[5];
ry(-3.063163879537735) q[2];
ry(-0.6872686570417059) q[5];
cx q[2],q[5];
ry(0.4559170213936716) q[3];
ry(-1.019831843648091) q[4];
cx q[3],q[4];
ry(1.5314344249041858) q[3];
ry(1.983291906253107) q[4];
cx q[3],q[4];
ry(-0.5921007702717682) q[4];
ry(2.6561160389015948) q[7];
cx q[4],q[7];
ry(1.3878009849046726) q[4];
ry(-2.680344725891447) q[7];
cx q[4],q[7];
ry(-2.010168002586622) q[5];
ry(-1.1076946167660973) q[6];
cx q[5],q[6];
ry(-3.1137379572990396) q[5];
ry(-1.9992219456240727) q[6];
cx q[5],q[6];
ry(0.06758184155724489) q[0];
ry(1.5525843036200468) q[1];
cx q[0],q[1];
ry(-1.973515809456785) q[0];
ry(-3.121928836321874) q[1];
cx q[0],q[1];
ry(2.103013684163208) q[2];
ry(1.6365587600947726) q[3];
cx q[2],q[3];
ry(-1.5487625163632845) q[2];
ry(1.010723846013216) q[3];
cx q[2],q[3];
ry(-1.550466761176135) q[4];
ry(0.802057679566703) q[5];
cx q[4],q[5];
ry(1.071952005773264) q[4];
ry(-1.1160823661085946) q[5];
cx q[4],q[5];
ry(1.3770752039435972) q[6];
ry(-2.3701586332543734) q[7];
cx q[6],q[7];
ry(2.221155635472252) q[6];
ry(3.0165244153571185) q[7];
cx q[6],q[7];
ry(2.1022779751373646) q[0];
ry(0.6056732950899953) q[2];
cx q[0],q[2];
ry(0.34067972691130727) q[0];
ry(2.4079541486594094) q[2];
cx q[0],q[2];
ry(1.0197255998900285) q[2];
ry(1.215682777678178) q[4];
cx q[2],q[4];
ry(-2.7622437334344014) q[2];
ry(-1.3040025960772226) q[4];
cx q[2],q[4];
ry(1.8374348185293572) q[4];
ry(2.6753574154848994) q[6];
cx q[4],q[6];
ry(1.2516931132514904) q[4];
ry(1.6887695013607213) q[6];
cx q[4],q[6];
ry(-2.657238334233345) q[1];
ry(0.5518387539539588) q[3];
cx q[1],q[3];
ry(1.4212059239456343) q[1];
ry(2.3840690138997345) q[3];
cx q[1],q[3];
ry(-1.2378462119889608) q[3];
ry(2.0370531025298293) q[5];
cx q[3],q[5];
ry(0.6735558659506831) q[3];
ry(2.6161953693463946) q[5];
cx q[3],q[5];
ry(-2.418915375398872) q[5];
ry(-1.6700987155596734) q[7];
cx q[5],q[7];
ry(0.544124958391025) q[5];
ry(-2.718018070563664) q[7];
cx q[5],q[7];
ry(-1.5064145293467766) q[0];
ry(-2.3815952697831966) q[3];
cx q[0],q[3];
ry(0.010170505918252226) q[0];
ry(-0.5861392852464763) q[3];
cx q[0],q[3];
ry(1.8120508008463272) q[1];
ry(-2.7271271379202946) q[2];
cx q[1],q[2];
ry(-2.7339586900458674) q[1];
ry(0.036124697406710204) q[2];
cx q[1],q[2];
ry(-2.1285513741104523) q[2];
ry(0.2036599745987327) q[5];
cx q[2],q[5];
ry(2.761178351712249) q[2];
ry(2.6763879913543165) q[5];
cx q[2],q[5];
ry(-2.9323583965119013) q[3];
ry(-0.39621076147099343) q[4];
cx q[3],q[4];
ry(2.2922087697351845) q[3];
ry(1.7001064452425212) q[4];
cx q[3],q[4];
ry(-2.7631374579362165) q[4];
ry(1.1578893479525352) q[7];
cx q[4],q[7];
ry(-3.1241380958254146) q[4];
ry(-0.29514393507089487) q[7];
cx q[4],q[7];
ry(1.7681920180698825) q[5];
ry(-1.757330178178459) q[6];
cx q[5],q[6];
ry(2.2911335090107485) q[5];
ry(-1.5531917087729268) q[6];
cx q[5],q[6];
ry(-0.3795133903412352) q[0];
ry(2.601288938511352) q[1];
cx q[0],q[1];
ry(0.034390214663309486) q[0];
ry(-2.496602062628263) q[1];
cx q[0],q[1];
ry(3.1042404581936123) q[2];
ry(-2.7466626282090267) q[3];
cx q[2],q[3];
ry(2.0611160279569916) q[2];
ry(0.5016107092913122) q[3];
cx q[2],q[3];
ry(-2.187271333059373) q[4];
ry(-0.022986594749620205) q[5];
cx q[4],q[5];
ry(1.3920650969373543) q[4];
ry(0.2920797945230129) q[5];
cx q[4],q[5];
ry(1.0251183297488815) q[6];
ry(-2.203802996271316) q[7];
cx q[6],q[7];
ry(0.4178600945106003) q[6];
ry(-1.1041597875372933) q[7];
cx q[6],q[7];
ry(1.7873961284712587) q[0];
ry(0.4465416132607327) q[2];
cx q[0],q[2];
ry(-3.1384550357130023) q[0];
ry(1.7931181787849124) q[2];
cx q[0],q[2];
ry(0.7101467038307814) q[2];
ry(2.8261105148474726) q[4];
cx q[2],q[4];
ry(2.0453672747143075) q[2];
ry(2.892703918364285) q[4];
cx q[2],q[4];
ry(2.3214746971056996) q[4];
ry(0.16983822581819374) q[6];
cx q[4],q[6];
ry(1.0406745603682217) q[4];
ry(2.925346599800888) q[6];
cx q[4],q[6];
ry(-1.3402892225442002) q[1];
ry(1.7361768434922362) q[3];
cx q[1],q[3];
ry(-2.9696838319306673) q[1];
ry(0.01977454968267396) q[3];
cx q[1],q[3];
ry(-2.626991524561639) q[3];
ry(-1.473594888178474) q[5];
cx q[3],q[5];
ry(-2.135987389687825) q[3];
ry(1.418853912523839) q[5];
cx q[3],q[5];
ry(-0.025500400391005762) q[5];
ry(-1.9251773060203035) q[7];
cx q[5],q[7];
ry(1.3681725215358171) q[5];
ry(-1.415507526125249) q[7];
cx q[5],q[7];
ry(2.6588170719568494) q[0];
ry(-2.037157219578389) q[3];
cx q[0],q[3];
ry(0.6651239291150546) q[0];
ry(-1.2201669707366871) q[3];
cx q[0],q[3];
ry(-2.5332150004601486) q[1];
ry(-1.1821522555795827) q[2];
cx q[1],q[2];
ry(-1.0781457640448346) q[1];
ry(-2.8823872469875376) q[2];
cx q[1],q[2];
ry(0.2968738795386585) q[2];
ry(0.986900368000363) q[5];
cx q[2],q[5];
ry(-1.3007318858174006) q[2];
ry(-0.4257741568836678) q[5];
cx q[2],q[5];
ry(0.12766284268648748) q[3];
ry(1.4748528754073922) q[4];
cx q[3],q[4];
ry(-1.3701858065512784) q[3];
ry(-1.4101674973691825) q[4];
cx q[3],q[4];
ry(-0.6191394904846668) q[4];
ry(-1.3736557989123712) q[7];
cx q[4],q[7];
ry(-1.5488749675459745) q[4];
ry(-2.9492939409863115) q[7];
cx q[4],q[7];
ry(2.17375410730401) q[5];
ry(-0.8615220338246914) q[6];
cx q[5],q[6];
ry(0.5966242202806497) q[5];
ry(2.6330348083749358) q[6];
cx q[5],q[6];
ry(-1.4632440090760928) q[0];
ry(0.014155901964385757) q[1];
cx q[0],q[1];
ry(-1.2178157215738645) q[0];
ry(-1.040554788303538) q[1];
cx q[0],q[1];
ry(1.0020143910047739) q[2];
ry(1.7659598030283856) q[3];
cx q[2],q[3];
ry(2.5562706293036217) q[2];
ry(1.8393771085849788) q[3];
cx q[2],q[3];
ry(-1.017696246915074) q[4];
ry(-0.7563384945496655) q[5];
cx q[4],q[5];
ry(-0.9133918101457175) q[4];
ry(-1.0314605177724763) q[5];
cx q[4],q[5];
ry(-1.4802889131297263) q[6];
ry(-1.959497606739392) q[7];
cx q[6],q[7];
ry(2.6429900142955804) q[6];
ry(-0.49104678846547833) q[7];
cx q[6],q[7];
ry(-2.281405091051651) q[0];
ry(2.38859533957349) q[2];
cx q[0],q[2];
ry(-0.19778945837382123) q[0];
ry(0.6001659410086582) q[2];
cx q[0],q[2];
ry(-2.4874172720806813) q[2];
ry(2.1187681579794133) q[4];
cx q[2],q[4];
ry(0.3074865234364106) q[2];
ry(-2.489621830449137) q[4];
cx q[2],q[4];
ry(1.0722094112592109) q[4];
ry(-2.229079424439718) q[6];
cx q[4],q[6];
ry(1.906996525638248) q[4];
ry(0.6156526693522578) q[6];
cx q[4],q[6];
ry(-2.699371185384178) q[1];
ry(1.6919493237835317) q[3];
cx q[1],q[3];
ry(1.8550984313543433) q[1];
ry(-0.014539541605519282) q[3];
cx q[1],q[3];
ry(2.497518309098741) q[3];
ry(-1.8546202175891702) q[5];
cx q[3],q[5];
ry(-0.8164715554497026) q[3];
ry(1.5712688092106433) q[5];
cx q[3],q[5];
ry(-1.4792768654882504) q[5];
ry(-1.010872741622884) q[7];
cx q[5],q[7];
ry(2.5785331507505767) q[5];
ry(2.663011858090556) q[7];
cx q[5],q[7];
ry(0.7884190770271495) q[0];
ry(0.9453381061513187) q[3];
cx q[0],q[3];
ry(0.7151306116650071) q[0];
ry(0.12509184388344874) q[3];
cx q[0],q[3];
ry(2.978364205043773) q[1];
ry(-1.9458368115412492) q[2];
cx q[1],q[2];
ry(0.24947304082926278) q[1];
ry(2.1006735064852475) q[2];
cx q[1],q[2];
ry(-0.27508568200395356) q[2];
ry(0.8324012937545602) q[5];
cx q[2],q[5];
ry(-1.5165064599419444) q[2];
ry(-3.0677605740014378) q[5];
cx q[2],q[5];
ry(-0.3598573521586861) q[3];
ry(-1.0196021230072854) q[4];
cx q[3],q[4];
ry(-1.1021259989132757) q[3];
ry(2.2259393595395665) q[4];
cx q[3],q[4];
ry(0.09891692745092005) q[4];
ry(3.0771771988613166) q[7];
cx q[4],q[7];
ry(2.2207993069768923) q[4];
ry(-0.530308682968876) q[7];
cx q[4],q[7];
ry(-0.34076316862884865) q[5];
ry(1.3518157355583842) q[6];
cx q[5],q[6];
ry(2.785701673655196) q[5];
ry(2.0518395119883053) q[6];
cx q[5],q[6];
ry(-1.961931163408335) q[0];
ry(-0.29037858370243974) q[1];
cx q[0],q[1];
ry(-2.992374903865902) q[0];
ry(1.121885085333066) q[1];
cx q[0],q[1];
ry(-2.127198563944109) q[2];
ry(1.7198750449671996) q[3];
cx q[2],q[3];
ry(1.5954668558990415) q[2];
ry(-0.44537527182316233) q[3];
cx q[2],q[3];
ry(2.4352006154497645) q[4];
ry(2.1871535624918383) q[5];
cx q[4],q[5];
ry(0.915678651537851) q[4];
ry(2.788366275071931) q[5];
cx q[4],q[5];
ry(-3.108456726656628) q[6];
ry(-2.2870149198694425) q[7];
cx q[6],q[7];
ry(2.596745088985552) q[6];
ry(-1.6810286427691379) q[7];
cx q[6],q[7];
ry(0.5981527522815231) q[0];
ry(-2.014232226485646) q[2];
cx q[0],q[2];
ry(-1.4489607090880483) q[0];
ry(0.6968471311053478) q[2];
cx q[0],q[2];
ry(-3.0319554003201086) q[2];
ry(-0.4184071405284362) q[4];
cx q[2],q[4];
ry(-1.5290170156730492) q[2];
ry(-0.2328018067101221) q[4];
cx q[2],q[4];
ry(-0.7414076013812522) q[4];
ry(3.070078821971494) q[6];
cx q[4],q[6];
ry(0.6538410214559747) q[4];
ry(0.34637589406454333) q[6];
cx q[4],q[6];
ry(-1.9641127393814024) q[1];
ry(1.8874176020580427) q[3];
cx q[1],q[3];
ry(0.4195329141093298) q[1];
ry(-0.27550765028923935) q[3];
cx q[1],q[3];
ry(-1.323263061843261) q[3];
ry(2.066780697931054) q[5];
cx q[3],q[5];
ry(1.4657204845901441) q[3];
ry(-0.7182378607147362) q[5];
cx q[3],q[5];
ry(-0.4567000059681936) q[5];
ry(1.0457065785925375) q[7];
cx q[5],q[7];
ry(0.6748142375869577) q[5];
ry(-1.0288403679732843) q[7];
cx q[5],q[7];
ry(-2.738958589083702) q[0];
ry(-2.9295118389510932) q[3];
cx q[0],q[3];
ry(-2.7861106762295798) q[0];
ry(0.23436562907597627) q[3];
cx q[0],q[3];
ry(-1.515883793114174) q[1];
ry(-2.1073498887362927) q[2];
cx q[1],q[2];
ry(-2.220626123250501) q[1];
ry(0.9025096726754662) q[2];
cx q[1],q[2];
ry(0.37242878187670847) q[2];
ry(1.892477812103144) q[5];
cx q[2],q[5];
ry(-2.8589197208234696) q[2];
ry(2.077026227821042) q[5];
cx q[2],q[5];
ry(0.8985249340822172) q[3];
ry(-0.10546153708578776) q[4];
cx q[3],q[4];
ry(-0.6687819553597234) q[3];
ry(-0.7766095917069702) q[4];
cx q[3],q[4];
ry(1.452922998112868) q[4];
ry(-1.0967209774891433) q[7];
cx q[4],q[7];
ry(-0.16793835021760817) q[4];
ry(0.009155853662625658) q[7];
cx q[4],q[7];
ry(2.3060198470580957) q[5];
ry(2.149272123706605) q[6];
cx q[5],q[6];
ry(-1.5482608180919735) q[5];
ry(1.1086984329143148) q[6];
cx q[5],q[6];
ry(-1.692898129773928) q[0];
ry(-0.6596203739181865) q[1];
cx q[0],q[1];
ry(1.8739514596788542) q[0];
ry(-2.029426305572672) q[1];
cx q[0],q[1];
ry(-0.45447549520351044) q[2];
ry(0.6879156255257658) q[3];
cx q[2],q[3];
ry(1.1868854041717212) q[2];
ry(-0.3984305547237694) q[3];
cx q[2],q[3];
ry(-1.3209082338355744) q[4];
ry(0.26065750572349355) q[5];
cx q[4],q[5];
ry(0.3674784754350191) q[4];
ry(-0.7696874332247792) q[5];
cx q[4],q[5];
ry(-1.3551683230948726) q[6];
ry(-0.593792327930954) q[7];
cx q[6],q[7];
ry(0.4285785947907905) q[6];
ry(2.4793305694530057) q[7];
cx q[6],q[7];
ry(2.16234333130806) q[0];
ry(2.372982150584008) q[2];
cx q[0],q[2];
ry(2.5325516475483103) q[0];
ry(-1.248972159296665) q[2];
cx q[0],q[2];
ry(1.5395614367624177) q[2];
ry(-1.6525205382489316) q[4];
cx q[2],q[4];
ry(-2.625435945468931) q[2];
ry(-2.8897641986038214) q[4];
cx q[2],q[4];
ry(1.2589533054036943) q[4];
ry(2.352301096224813) q[6];
cx q[4],q[6];
ry(-0.44015031086190276) q[4];
ry(-1.15239439769594) q[6];
cx q[4],q[6];
ry(0.5879566418309023) q[1];
ry(-0.747780581045692) q[3];
cx q[1],q[3];
ry(1.0951671142108288) q[1];
ry(-2.2608879791764904) q[3];
cx q[1],q[3];
ry(0.1902131866206632) q[3];
ry(-2.969943857249501) q[5];
cx q[3],q[5];
ry(-1.6748678122609792) q[3];
ry(1.2284524112007258) q[5];
cx q[3],q[5];
ry(-1.6974843417146164) q[5];
ry(-2.525303620454553) q[7];
cx q[5],q[7];
ry(1.9155207342936142) q[5];
ry(2.8768647312625064) q[7];
cx q[5],q[7];
ry(-0.04434073501961091) q[0];
ry(-2.7208986692695) q[3];
cx q[0],q[3];
ry(0.0470331298068297) q[0];
ry(2.257008840576871) q[3];
cx q[0],q[3];
ry(1.564926882386569) q[1];
ry(2.5770955279780945) q[2];
cx q[1],q[2];
ry(-1.25600219032383) q[1];
ry(-3.0719370956948935) q[2];
cx q[1],q[2];
ry(-2.7008069386469296) q[2];
ry(-1.9276491127987407) q[5];
cx q[2],q[5];
ry(0.05489101101192171) q[2];
ry(0.14355017020507344) q[5];
cx q[2],q[5];
ry(-1.2529638544415551) q[3];
ry(-1.2471211387988994) q[4];
cx q[3],q[4];
ry(-0.68254194006962) q[3];
ry(1.6560114366019187) q[4];
cx q[3],q[4];
ry(-0.5001130245496392) q[4];
ry(1.5864399091675034) q[7];
cx q[4],q[7];
ry(0.16539451291857174) q[4];
ry(2.9690599489623026) q[7];
cx q[4],q[7];
ry(1.7358089071676899) q[5];
ry(1.226811390143462) q[6];
cx q[5],q[6];
ry(-0.8614184017468585) q[5];
ry(3.0119400749742407) q[6];
cx q[5],q[6];
ry(1.2816310152877615) q[0];
ry(-1.9195313100027755) q[1];
cx q[0],q[1];
ry(-2.4514198916071197) q[0];
ry(-1.4109625315731003) q[1];
cx q[0],q[1];
ry(-2.730977236579157) q[2];
ry(-2.486859157384233) q[3];
cx q[2],q[3];
ry(-2.0761154201699847) q[2];
ry(2.4887467254488325) q[3];
cx q[2],q[3];
ry(2.187037510298583) q[4];
ry(1.7128008416503804) q[5];
cx q[4],q[5];
ry(-0.5715144248765495) q[4];
ry(1.4453696221058427) q[5];
cx q[4],q[5];
ry(-2.882215135164132) q[6];
ry(2.6511395400709117) q[7];
cx q[6],q[7];
ry(-1.8014802423012783) q[6];
ry(-1.6646751747229152) q[7];
cx q[6],q[7];
ry(2.9628287487627967) q[0];
ry(-2.6641520854392087) q[2];
cx q[0],q[2];
ry(-2.157538016853252) q[0];
ry(-0.7649284009111386) q[2];
cx q[0],q[2];
ry(1.7043770702823142) q[2];
ry(-1.4842249437436408) q[4];
cx q[2],q[4];
ry(-1.2134038655548152) q[2];
ry(-2.374500239553155) q[4];
cx q[2],q[4];
ry(-1.6304451788629581) q[4];
ry(-2.6305522152384624) q[6];
cx q[4],q[6];
ry(-0.9361095012144021) q[4];
ry(-2.751006731090172) q[6];
cx q[4],q[6];
ry(-0.46537682723768486) q[1];
ry(-0.5817613816219466) q[3];
cx q[1],q[3];
ry(-0.7307870607990852) q[1];
ry(2.506131293610308) q[3];
cx q[1],q[3];
ry(2.8679642471047218) q[3];
ry(1.7014531769451615) q[5];
cx q[3],q[5];
ry(1.738074715330206) q[3];
ry(-1.6761759535718337) q[5];
cx q[3],q[5];
ry(0.6532427054785656) q[5];
ry(-0.9909435829866053) q[7];
cx q[5],q[7];
ry(-0.5274379966953003) q[5];
ry(0.35697796645394986) q[7];
cx q[5],q[7];
ry(1.417579992881815) q[0];
ry(-0.06536467623643405) q[3];
cx q[0],q[3];
ry(-1.3481219941759202) q[0];
ry(0.25875172413769754) q[3];
cx q[0],q[3];
ry(2.0128975965362566) q[1];
ry(0.9507828652588409) q[2];
cx q[1],q[2];
ry(-1.950770593304644) q[1];
ry(2.325363009031645) q[2];
cx q[1],q[2];
ry(-2.8378230638905215) q[2];
ry(0.031045886608297157) q[5];
cx q[2],q[5];
ry(-0.7195940906472008) q[2];
ry(0.3751791757724434) q[5];
cx q[2],q[5];
ry(2.7372275628939) q[3];
ry(-2.9393236747434512) q[4];
cx q[3],q[4];
ry(-0.40673943816383945) q[3];
ry(-2.6132216300128053) q[4];
cx q[3],q[4];
ry(1.6056853578730206) q[4];
ry(1.373589122483583) q[7];
cx q[4],q[7];
ry(0.8106673732684216) q[4];
ry(-2.2013342528727247) q[7];
cx q[4],q[7];
ry(-1.0819959910962327) q[5];
ry(0.09524586355824471) q[6];
cx q[5],q[6];
ry(-0.6934203238091678) q[5];
ry(-0.2509125195287778) q[6];
cx q[5],q[6];
ry(-0.5911425348735939) q[0];
ry(1.6077030420990426) q[1];
cx q[0],q[1];
ry(2.1740380340386825) q[0];
ry(1.6142185201403645) q[1];
cx q[0],q[1];
ry(-1.735181807194388) q[2];
ry(-2.417422201944733) q[3];
cx q[2],q[3];
ry(0.09010750250900094) q[2];
ry(2.2431379418370447) q[3];
cx q[2],q[3];
ry(-1.0115502438870845) q[4];
ry(-2.4990885461474908) q[5];
cx q[4],q[5];
ry(0.18629598110039947) q[4];
ry(1.9114889871584175) q[5];
cx q[4],q[5];
ry(1.7562132441252682) q[6];
ry(0.8124935789490824) q[7];
cx q[6],q[7];
ry(0.6068638807359634) q[6];
ry(-0.9282166687528894) q[7];
cx q[6],q[7];
ry(-2.4440969186115695) q[0];
ry(2.021355879126705) q[2];
cx q[0],q[2];
ry(-1.5710953577977875) q[0];
ry(1.7314703933441011) q[2];
cx q[0],q[2];
ry(1.1886862840640537) q[2];
ry(0.847832513577572) q[4];
cx q[2],q[4];
ry(-2.7353511302834956) q[2];
ry(0.7222121302202775) q[4];
cx q[2],q[4];
ry(1.570433093652847) q[4];
ry(-0.16562647987618975) q[6];
cx q[4],q[6];
ry(2.4488600560549165) q[4];
ry(-2.551880085624814) q[6];
cx q[4],q[6];
ry(-0.6372892284384488) q[1];
ry(2.0671723242699755) q[3];
cx q[1],q[3];
ry(-0.8731293986435569) q[1];
ry(2.901023020120677) q[3];
cx q[1],q[3];
ry(2.7983214506840532) q[3];
ry(-0.39673707213041265) q[5];
cx q[3],q[5];
ry(-2.6468485942641364) q[3];
ry(-1.1688796068779441) q[5];
cx q[3],q[5];
ry(-0.7952520154476588) q[5];
ry(0.9171345193532838) q[7];
cx q[5],q[7];
ry(2.777567346961001) q[5];
ry(1.8355167935190648) q[7];
cx q[5],q[7];
ry(1.920135681038012) q[0];
ry(-0.4858544033876974) q[3];
cx q[0],q[3];
ry(0.08598094213073892) q[0];
ry(-2.501255136245042) q[3];
cx q[0],q[3];
ry(-0.9415562519979153) q[1];
ry(-0.619111001537587) q[2];
cx q[1],q[2];
ry(-2.2239372584229398) q[1];
ry(2.365890982083423) q[2];
cx q[1],q[2];
ry(-2.10536822842949) q[2];
ry(-0.7744381295953385) q[5];
cx q[2],q[5];
ry(-0.9524900919009047) q[2];
ry(-1.4767517266378158) q[5];
cx q[2],q[5];
ry(-2.2772808857292826) q[3];
ry(2.42211484300364) q[4];
cx q[3],q[4];
ry(1.106064271225146) q[3];
ry(-1.6097512141069166) q[4];
cx q[3],q[4];
ry(0.6292685950929156) q[4];
ry(2.3560186160925185) q[7];
cx q[4],q[7];
ry(-2.7327801168758556) q[4];
ry(1.733166810634557) q[7];
cx q[4],q[7];
ry(2.9304109147404533) q[5];
ry(2.223366991097241) q[6];
cx q[5],q[6];
ry(-1.2522737548852663) q[5];
ry(0.9282362761913152) q[6];
cx q[5],q[6];
ry(1.7033504038198628) q[0];
ry(1.7185717764850992) q[1];
cx q[0],q[1];
ry(-0.5691428596024624) q[0];
ry(1.8805683805558493) q[1];
cx q[0],q[1];
ry(-0.42094260324243327) q[2];
ry(0.12856564789924735) q[3];
cx q[2],q[3];
ry(-0.015286472164706133) q[2];
ry(0.8249809015969225) q[3];
cx q[2],q[3];
ry(-1.2367817619449382) q[4];
ry(-0.9534577876814012) q[5];
cx q[4],q[5];
ry(1.9030811053335424) q[4];
ry(-2.0870148872719927) q[5];
cx q[4],q[5];
ry(-2.281143104622927) q[6];
ry(0.4834561848857186) q[7];
cx q[6],q[7];
ry(0.2845728271441697) q[6];
ry(0.7530040799170532) q[7];
cx q[6],q[7];
ry(-2.732548235552162) q[0];
ry(-2.526069946141589) q[2];
cx q[0],q[2];
ry(-1.1361061174679634) q[0];
ry(-0.8160412004913074) q[2];
cx q[0],q[2];
ry(-2.2034213293261296) q[2];
ry(1.5612291178203375) q[4];
cx q[2],q[4];
ry(0.40102733849878097) q[2];
ry(-0.4553252794306676) q[4];
cx q[2],q[4];
ry(1.7588862774193732) q[4];
ry(0.09384761821310866) q[6];
cx q[4],q[6];
ry(1.6319090407471666) q[4];
ry(1.499248826392078) q[6];
cx q[4],q[6];
ry(-1.9864651068455914) q[1];
ry(2.5335689342715093) q[3];
cx q[1],q[3];
ry(0.5247921192646706) q[1];
ry(1.1313624613520918) q[3];
cx q[1],q[3];
ry(-2.626430343901003) q[3];
ry(-3.0448159528024457) q[5];
cx q[3],q[5];
ry(-1.2146703972512558) q[3];
ry(0.3673136521741762) q[5];
cx q[3],q[5];
ry(2.5247213245830236) q[5];
ry(1.6686531200090045) q[7];
cx q[5],q[7];
ry(1.3131513926155893) q[5];
ry(-0.5942968613704634) q[7];
cx q[5],q[7];
ry(-1.5949041613334745) q[0];
ry(0.3431136426962178) q[3];
cx q[0],q[3];
ry(-0.8208868615856265) q[0];
ry(-0.5763471146801837) q[3];
cx q[0],q[3];
ry(-0.5882707087733205) q[1];
ry(3.073855697597532) q[2];
cx q[1],q[2];
ry(-0.5652613302820997) q[1];
ry(2.5871649747231356) q[2];
cx q[1],q[2];
ry(-2.1162712444061063) q[2];
ry(3.1002727991297117) q[5];
cx q[2],q[5];
ry(1.4052770311532996) q[2];
ry(-0.33258718144956134) q[5];
cx q[2],q[5];
ry(2.5518242905629362) q[3];
ry(-2.6777511843071213) q[4];
cx q[3],q[4];
ry(0.6521353212502943) q[3];
ry(-0.7962859383737345) q[4];
cx q[3],q[4];
ry(2.183936759353598) q[4];
ry(-0.6945045431392494) q[7];
cx q[4],q[7];
ry(-2.630626193666543) q[4];
ry(-2.1902659915979728) q[7];
cx q[4],q[7];
ry(-0.12658233443475986) q[5];
ry(3.095162692066149) q[6];
cx q[5],q[6];
ry(2.0253793682401473) q[5];
ry(-1.3361596683600079) q[6];
cx q[5],q[6];
ry(-0.056316382385356974) q[0];
ry(0.04893111641748149) q[1];
cx q[0],q[1];
ry(-2.8658775358768387) q[0];
ry(0.1974933994593062) q[1];
cx q[0],q[1];
ry(-1.6480855248473203) q[2];
ry(3.1277219491858665) q[3];
cx q[2],q[3];
ry(0.38482161823522265) q[2];
ry(-1.5430435443134423) q[3];
cx q[2],q[3];
ry(2.1629051225312907) q[4];
ry(-3.1409218616297743) q[5];
cx q[4],q[5];
ry(-2.137409670863055) q[4];
ry(-1.6883162003309344) q[5];
cx q[4],q[5];
ry(0.16033028354290876) q[6];
ry(-2.906353588039333) q[7];
cx q[6],q[7];
ry(2.450345988651617) q[6];
ry(-2.7495003341794373) q[7];
cx q[6],q[7];
ry(-1.3930663951355111) q[0];
ry(-3.088952056234847) q[2];
cx q[0],q[2];
ry(-1.3751492561759635) q[0];
ry(-0.1457800230716213) q[2];
cx q[0],q[2];
ry(-2.622615696378159) q[2];
ry(-0.4765080074922384) q[4];
cx q[2],q[4];
ry(-0.8788270866302001) q[2];
ry(-2.9534317821086096) q[4];
cx q[2],q[4];
ry(-0.26259207386807637) q[4];
ry(-2.993457919207914) q[6];
cx q[4],q[6];
ry(0.22180408481496805) q[4];
ry(-0.34631474589384037) q[6];
cx q[4],q[6];
ry(1.2803838256193483) q[1];
ry(1.8915187035043224) q[3];
cx q[1],q[3];
ry(2.65004951917637) q[1];
ry(-2.031567114144405) q[3];
cx q[1],q[3];
ry(1.6544984123026119) q[3];
ry(-1.1554768389337886) q[5];
cx q[3],q[5];
ry(-1.7631580891899787) q[3];
ry(2.407704750383111) q[5];
cx q[3],q[5];
ry(1.3965687587219735) q[5];
ry(-2.275802446029623) q[7];
cx q[5],q[7];
ry(1.0357991191144733) q[5];
ry(-2.643669952531132) q[7];
cx q[5],q[7];
ry(0.9994121923766858) q[0];
ry(1.917549149546579) q[3];
cx q[0],q[3];
ry(0.15254893463161157) q[0];
ry(-1.618210806507647) q[3];
cx q[0],q[3];
ry(-2.785732156635594) q[1];
ry(1.0221095238493856) q[2];
cx q[1],q[2];
ry(1.7772501086961225) q[1];
ry(-0.8580026515070858) q[2];
cx q[1],q[2];
ry(2.5538124615860855) q[2];
ry(0.668437802547487) q[5];
cx q[2],q[5];
ry(1.808538123911533) q[2];
ry(-2.916744804559883) q[5];
cx q[2],q[5];
ry(1.2747509572979716) q[3];
ry(1.064192825808635) q[4];
cx q[3],q[4];
ry(1.73691591810372) q[3];
ry(-1.5513253640959999) q[4];
cx q[3],q[4];
ry(-2.750635370366185) q[4];
ry(0.20775110745156877) q[7];
cx q[4],q[7];
ry(2.709405206267181) q[4];
ry(0.570517110101384) q[7];
cx q[4],q[7];
ry(-2.1979754739880546) q[5];
ry(2.9136968969562953) q[6];
cx q[5],q[6];
ry(0.11372590278726391) q[5];
ry(0.3425138015687814) q[6];
cx q[5],q[6];
ry(0.985792568607221) q[0];
ry(-0.5507317286877624) q[1];
cx q[0],q[1];
ry(-1.8411141379819402) q[0];
ry(0.03512686944577862) q[1];
cx q[0],q[1];
ry(2.5366274609352772) q[2];
ry(1.4303914651653515) q[3];
cx q[2],q[3];
ry(-2.3643118651515977) q[2];
ry(1.3834829049397213) q[3];
cx q[2],q[3];
ry(0.09222573288491911) q[4];
ry(0.6271780261268989) q[5];
cx q[4],q[5];
ry(1.4334764885199733) q[4];
ry(1.1082311588475022) q[5];
cx q[4],q[5];
ry(2.9335609213008835) q[6];
ry(-1.5153099689010112) q[7];
cx q[6],q[7];
ry(-0.90958889007619) q[6];
ry(0.09332219274710987) q[7];
cx q[6],q[7];
ry(-1.330710178312507) q[0];
ry(1.6881968315455889) q[2];
cx q[0],q[2];
ry(-1.9307410839030636) q[0];
ry(0.03014569700624732) q[2];
cx q[0],q[2];
ry(-0.3256591446543354) q[2];
ry(-0.4857452089875268) q[4];
cx q[2],q[4];
ry(1.3905525355952553) q[2];
ry(1.9830690194014338) q[4];
cx q[2],q[4];
ry(-2.4301883182364103) q[4];
ry(-2.2239019784006095) q[6];
cx q[4],q[6];
ry(-1.8315350617015123) q[4];
ry(0.9046290714020646) q[6];
cx q[4],q[6];
ry(-2.533680565568797) q[1];
ry(1.6151987643940586) q[3];
cx q[1],q[3];
ry(0.8475054856130149) q[1];
ry(-0.9794391622761048) q[3];
cx q[1],q[3];
ry(1.9574010150499739) q[3];
ry(-0.8969296744641291) q[5];
cx q[3],q[5];
ry(-1.5474774549582158) q[3];
ry(-1.7578406090538732) q[5];
cx q[3],q[5];
ry(2.5266695200080678) q[5];
ry(2.9212022314355046) q[7];
cx q[5],q[7];
ry(3.113481885881676) q[5];
ry(-1.092431879729389) q[7];
cx q[5],q[7];
ry(-1.6168040150611658) q[0];
ry(2.94928805987926) q[3];
cx q[0],q[3];
ry(-0.23758993035720355) q[0];
ry(-1.118653120953117) q[3];
cx q[0],q[3];
ry(-1.8628359772696645) q[1];
ry(2.464765478714428) q[2];
cx q[1],q[2];
ry(-2.624005658468533) q[1];
ry(-2.6167135811037667) q[2];
cx q[1],q[2];
ry(-2.658795887309047) q[2];
ry(-2.047809679833322) q[5];
cx q[2],q[5];
ry(0.6239370728168154) q[2];
ry(1.0283875629792183) q[5];
cx q[2],q[5];
ry(-0.49920858497759024) q[3];
ry(-2.18858704470422) q[4];
cx q[3],q[4];
ry(-1.8706193665407143) q[3];
ry(-2.754022118080495) q[4];
cx q[3],q[4];
ry(-2.0590588883404872) q[4];
ry(1.139266170660861) q[7];
cx q[4],q[7];
ry(0.5018709289540659) q[4];
ry(-0.7256578382631332) q[7];
cx q[4],q[7];
ry(-2.9046323793044806) q[5];
ry(-2.075472021775921) q[6];
cx q[5],q[6];
ry(-1.2586765751276596) q[5];
ry(-1.036794836651052) q[6];
cx q[5],q[6];
ry(-0.33996786313223293) q[0];
ry(-1.8384849863943629) q[1];
cx q[0],q[1];
ry(-2.0726948779874252) q[0];
ry(1.8505813227981145) q[1];
cx q[0],q[1];
ry(2.0434559141895434) q[2];
ry(-2.0754679527476076) q[3];
cx q[2],q[3];
ry(2.582056141369785) q[2];
ry(0.0858929829937134) q[3];
cx q[2],q[3];
ry(-0.2510511620385829) q[4];
ry(0.7832395120200806) q[5];
cx q[4],q[5];
ry(2.4100243269961568) q[4];
ry(-2.151790070732855) q[5];
cx q[4],q[5];
ry(0.44884271900356953) q[6];
ry(0.6970832787426986) q[7];
cx q[6],q[7];
ry(1.2270168129122974) q[6];
ry(2.0319131018308365) q[7];
cx q[6],q[7];
ry(0.9963171797409016) q[0];
ry(0.2818231913980168) q[2];
cx q[0],q[2];
ry(-1.288962132706196) q[0];
ry(2.725480703061422) q[2];
cx q[0],q[2];
ry(1.0920630989966513) q[2];
ry(-2.5315995121882415) q[4];
cx q[2],q[4];
ry(-2.0622344255335006) q[2];
ry(2.1274120419005698) q[4];
cx q[2],q[4];
ry(-1.0219867659924864) q[4];
ry(1.5200888828443486) q[6];
cx q[4],q[6];
ry(0.509983169023938) q[4];
ry(-0.4073282164837816) q[6];
cx q[4],q[6];
ry(-1.277149085571784) q[1];
ry(-2.8637126148896073) q[3];
cx q[1],q[3];
ry(1.658239060122829) q[1];
ry(-1.9207250421881203) q[3];
cx q[1],q[3];
ry(-2.5966513139626684) q[3];
ry(-0.11207146365881826) q[5];
cx q[3],q[5];
ry(2.251772545574169) q[3];
ry(0.9224743483254768) q[5];
cx q[3],q[5];
ry(1.953237050092942) q[5];
ry(-1.956967862420111) q[7];
cx q[5],q[7];
ry(-3.066749320352135) q[5];
ry(-2.1646594354966044) q[7];
cx q[5],q[7];
ry(0.7000587266025304) q[0];
ry(-1.112933875664382) q[3];
cx q[0],q[3];
ry(-2.790829095795045) q[0];
ry(-1.067748343106996) q[3];
cx q[0],q[3];
ry(2.614234996616718) q[1];
ry(1.2728090074185827) q[2];
cx q[1],q[2];
ry(-0.7678740365181511) q[1];
ry(-2.4424552838852023) q[2];
cx q[1],q[2];
ry(-1.2704511332018988) q[2];
ry(2.649492120303392) q[5];
cx q[2],q[5];
ry(2.7647296893986693) q[2];
ry(-0.49555071183317434) q[5];
cx q[2],q[5];
ry(-3.136873147854465) q[3];
ry(1.8286351063769957) q[4];
cx q[3],q[4];
ry(-0.21094885728313903) q[3];
ry(0.956352269041091) q[4];
cx q[3],q[4];
ry(2.6573161547499415) q[4];
ry(-0.8888497315870482) q[7];
cx q[4],q[7];
ry(-1.2299754653028216) q[4];
ry(2.0869671240435617) q[7];
cx q[4],q[7];
ry(-1.3074979471295969) q[5];
ry(-1.1590219358876184) q[6];
cx q[5],q[6];
ry(-0.7742158987456733) q[5];
ry(-0.7598279301009407) q[6];
cx q[5],q[6];
ry(-0.9780615148303875) q[0];
ry(2.0956371443931596) q[1];
cx q[0],q[1];
ry(1.786408100148684) q[0];
ry(-1.618419578345975) q[1];
cx q[0],q[1];
ry(-3.120546686161869) q[2];
ry(3.057340677390349) q[3];
cx q[2],q[3];
ry(0.4873939453583658) q[2];
ry(-0.11708669125681799) q[3];
cx q[2],q[3];
ry(0.9864518029224411) q[4];
ry(-1.8771679356987834) q[5];
cx q[4],q[5];
ry(-0.06433143960841657) q[4];
ry(0.6470037540460236) q[5];
cx q[4],q[5];
ry(2.814029890325605) q[6];
ry(1.3546195192699553) q[7];
cx q[6],q[7];
ry(-2.0160329921704925) q[6];
ry(2.1018921924124427) q[7];
cx q[6],q[7];
ry(-0.298237880377636) q[0];
ry(-3.093520948138092) q[2];
cx q[0],q[2];
ry(1.1790717329805807) q[0];
ry(-1.1076396050935322) q[2];
cx q[0],q[2];
ry(0.43089014565130596) q[2];
ry(-0.7930086130781355) q[4];
cx q[2],q[4];
ry(1.95233685439389) q[2];
ry(-2.579582836508073) q[4];
cx q[2],q[4];
ry(-1.2223949008670048) q[4];
ry(0.23102990789010514) q[6];
cx q[4],q[6];
ry(-2.257701038469206) q[4];
ry(2.030554124123859) q[6];
cx q[4],q[6];
ry(-3.080113780853627) q[1];
ry(-1.1344688040051896) q[3];
cx q[1],q[3];
ry(0.6036207843773749) q[1];
ry(-1.7301852087457297) q[3];
cx q[1],q[3];
ry(-2.4253321508399877) q[3];
ry(-0.4251070780076433) q[5];
cx q[3],q[5];
ry(0.42218641424490677) q[3];
ry(0.9690440381830372) q[5];
cx q[3],q[5];
ry(-2.4877808504764705) q[5];
ry(0.3868223825083558) q[7];
cx q[5],q[7];
ry(-2.0647204217395627) q[5];
ry(0.028368118557380306) q[7];
cx q[5],q[7];
ry(-2.9813432407825937) q[0];
ry(1.841674247586318) q[3];
cx q[0],q[3];
ry(2.8295393645399343) q[0];
ry(-3.000102637724197) q[3];
cx q[0],q[3];
ry(-2.481378436082698) q[1];
ry(1.3743740200611647) q[2];
cx q[1],q[2];
ry(0.8622439752604137) q[1];
ry(-1.1019462623888927) q[2];
cx q[1],q[2];
ry(2.600500699998024) q[2];
ry(1.614932246746579) q[5];
cx q[2],q[5];
ry(0.5617486025231848) q[2];
ry(-0.9365208974909779) q[5];
cx q[2],q[5];
ry(1.8440130115827795) q[3];
ry(1.7468294421324992) q[4];
cx q[3],q[4];
ry(2.1968291321583355) q[3];
ry(-0.2320037246312232) q[4];
cx q[3],q[4];
ry(0.15150623686925557) q[4];
ry(2.601690823952876) q[7];
cx q[4],q[7];
ry(1.441032201610203) q[4];
ry(-0.8472164702412242) q[7];
cx q[4],q[7];
ry(-0.08171712863101593) q[5];
ry(-3.0362526484106653) q[6];
cx q[5],q[6];
ry(0.5691758921987145) q[5];
ry(-2.0751489106944385) q[6];
cx q[5],q[6];
ry(0.03520503495815852) q[0];
ry(0.8163898085164175) q[1];
cx q[0],q[1];
ry(-1.685510454267404) q[0];
ry(-2.532595124323728) q[1];
cx q[0],q[1];
ry(-0.14691363576022456) q[2];
ry(-0.3226688972145668) q[3];
cx q[2],q[3];
ry(1.9681840194101823) q[2];
ry(-0.5916181436715986) q[3];
cx q[2],q[3];
ry(-0.6026591879942478) q[4];
ry(0.5922801220105001) q[5];
cx q[4],q[5];
ry(2.2116193002096907) q[4];
ry(-2.345578616945993) q[5];
cx q[4],q[5];
ry(2.6568635610804234) q[6];
ry(0.8988510523794202) q[7];
cx q[6],q[7];
ry(-2.5764811676018717) q[6];
ry(-0.9578424867505078) q[7];
cx q[6],q[7];
ry(1.793859359483549) q[0];
ry(-0.713460700696932) q[2];
cx q[0],q[2];
ry(-2.7001163281084875) q[0];
ry(-0.16705924957217014) q[2];
cx q[0],q[2];
ry(0.009365580266277838) q[2];
ry(0.04968121551174784) q[4];
cx q[2],q[4];
ry(-0.1017766744668629) q[2];
ry(2.490181776879979) q[4];
cx q[2],q[4];
ry(1.6326998647514115) q[4];
ry(0.6100068335596462) q[6];
cx q[4],q[6];
ry(-2.118479785818555) q[4];
ry(0.3485041873071566) q[6];
cx q[4],q[6];
ry(-2.4514338752703173) q[1];
ry(-0.02525643812275486) q[3];
cx q[1],q[3];
ry(-2.105713145484641) q[1];
ry(0.8528537593759934) q[3];
cx q[1],q[3];
ry(2.7160141217205442) q[3];
ry(1.1984488741788124) q[5];
cx q[3],q[5];
ry(-1.48020426238367) q[3];
ry(0.9658647657056321) q[5];
cx q[3],q[5];
ry(0.5874792730046652) q[5];
ry(2.401289615731265) q[7];
cx q[5],q[7];
ry(-3.0920356801512776) q[5];
ry(-2.0559511792507417) q[7];
cx q[5],q[7];
ry(0.03572546273418897) q[0];
ry(-2.0791203310626) q[3];
cx q[0],q[3];
ry(-2.125754086123827) q[0];
ry(2.8099869273957023) q[3];
cx q[0],q[3];
ry(-2.820947414047742) q[1];
ry(-2.701880998878933) q[2];
cx q[1],q[2];
ry(-1.6454313043276394) q[1];
ry(1.2848472053122277) q[2];
cx q[1],q[2];
ry(0.6135599186824898) q[2];
ry(0.47097712765132005) q[5];
cx q[2],q[5];
ry(-2.304819346852468) q[2];
ry(1.4098133483936115) q[5];
cx q[2],q[5];
ry(-1.2253885317922217) q[3];
ry(2.3517312510849404) q[4];
cx q[3],q[4];
ry(-1.9915604678929864) q[3];
ry(-1.9056031188510054) q[4];
cx q[3],q[4];
ry(-2.5149443585310225) q[4];
ry(-2.6398056281204645) q[7];
cx q[4],q[7];
ry(-2.3596736078172658) q[4];
ry(0.2162023930909207) q[7];
cx q[4],q[7];
ry(-0.4806159694482952) q[5];
ry(-1.7611222645891003) q[6];
cx q[5],q[6];
ry(0.26030844230900146) q[5];
ry(-0.8570661445139162) q[6];
cx q[5],q[6];
ry(-1.3815421309440885) q[0];
ry(0.12756876429052524) q[1];
cx q[0],q[1];
ry(-1.6759199378798548) q[0];
ry(2.9560827878210887) q[1];
cx q[0],q[1];
ry(2.915534980009427) q[2];
ry(2.8646576510679935) q[3];
cx q[2],q[3];
ry(1.3709187100846814) q[2];
ry(-1.640120004494685) q[3];
cx q[2],q[3];
ry(-1.3009117183398367) q[4];
ry(-1.9905851812839552) q[5];
cx q[4],q[5];
ry(-1.284899413846184) q[4];
ry(-2.9000378418771118) q[5];
cx q[4],q[5];
ry(-0.8596178333823401) q[6];
ry(2.878851335896152) q[7];
cx q[6],q[7];
ry(2.9490933714063154) q[6];
ry(2.659120107659013) q[7];
cx q[6],q[7];
ry(1.3652038949691274) q[0];
ry(2.0697452925605306) q[2];
cx q[0],q[2];
ry(-1.983056126448048) q[0];
ry(-2.4474650815983505) q[2];
cx q[0],q[2];
ry(-1.175050173385644) q[2];
ry(-1.776834408399817) q[4];
cx q[2],q[4];
ry(-1.9267724461809868) q[2];
ry(1.9784323531387134) q[4];
cx q[2],q[4];
ry(2.069499571266316) q[4];
ry(-1.9082878609652392) q[6];
cx q[4],q[6];
ry(0.8551954158980414) q[4];
ry(-0.8787305913648772) q[6];
cx q[4],q[6];
ry(2.352563064940526) q[1];
ry(-0.6016212008042663) q[3];
cx q[1],q[3];
ry(-2.254430091341353) q[1];
ry(-1.3481838744502168) q[3];
cx q[1],q[3];
ry(-1.8954683437852722) q[3];
ry(0.866793090510756) q[5];
cx q[3],q[5];
ry(-1.5591433042734426) q[3];
ry(-2.085987617117066) q[5];
cx q[3],q[5];
ry(2.5741946402053477) q[5];
ry(0.5928494964312788) q[7];
cx q[5],q[7];
ry(-1.7780903737306681) q[5];
ry(2.0078070350195008) q[7];
cx q[5],q[7];
ry(1.0050473178571024) q[0];
ry(-1.6738218590954732) q[3];
cx q[0],q[3];
ry(-2.6172760623196187) q[0];
ry(2.5544832941794136) q[3];
cx q[0],q[3];
ry(-2.059804763104066) q[1];
ry(-0.6921700882169209) q[2];
cx q[1],q[2];
ry(0.7133235668234111) q[1];
ry(0.0770085406381229) q[2];
cx q[1],q[2];
ry(2.7327167286499408) q[2];
ry(-2.6019695136774605) q[5];
cx q[2],q[5];
ry(-0.9834867943618542) q[2];
ry(2.2423445982133323) q[5];
cx q[2],q[5];
ry(2.861860345865705) q[3];
ry(0.5994630676495543) q[4];
cx q[3],q[4];
ry(-0.7295069316522235) q[3];
ry(0.2965381908906375) q[4];
cx q[3],q[4];
ry(0.15287260766851893) q[4];
ry(-2.994130446109123) q[7];
cx q[4],q[7];
ry(1.0289402170582733) q[4];
ry(-0.11371238221226707) q[7];
cx q[4],q[7];
ry(1.5355435850248589) q[5];
ry(2.781423248844978) q[6];
cx q[5],q[6];
ry(0.6826208168274119) q[5];
ry(1.3566969344851634) q[6];
cx q[5],q[6];
ry(-3.0784106363114163) q[0];
ry(-1.9894028232235779) q[1];
cx q[0],q[1];
ry(-0.8264533223140358) q[0];
ry(-2.4611455889370504) q[1];
cx q[0],q[1];
ry(1.3637320159518775) q[2];
ry(1.7989992584324759) q[3];
cx q[2],q[3];
ry(0.005031545632314227) q[2];
ry(-1.8548104395360592) q[3];
cx q[2],q[3];
ry(-0.5563088713560126) q[4];
ry(2.161321538076414) q[5];
cx q[4],q[5];
ry(-0.5000469441786288) q[4];
ry(1.94322722122467) q[5];
cx q[4],q[5];
ry(-2.4128098609900492) q[6];
ry(2.608174090709461) q[7];
cx q[6],q[7];
ry(1.4607474733017494) q[6];
ry(1.3913261824920937) q[7];
cx q[6],q[7];
ry(1.29180256876794) q[0];
ry(-0.12644901977345244) q[2];
cx q[0],q[2];
ry(-2.379094310591933) q[0];
ry(-1.2957688363590387) q[2];
cx q[0],q[2];
ry(-0.19963499010007038) q[2];
ry(2.7449659075559345) q[4];
cx q[2],q[4];
ry(1.1560831659123798) q[2];
ry(2.3365674099778944) q[4];
cx q[2],q[4];
ry(1.249604922919823) q[4];
ry(0.10207227642136966) q[6];
cx q[4],q[6];
ry(-1.9359436944115105) q[4];
ry(3.0140056567227207) q[6];
cx q[4],q[6];
ry(-0.8966943045807652) q[1];
ry(2.5901239420605435) q[3];
cx q[1],q[3];
ry(-2.007933797387307) q[1];
ry(-1.2027939971204689) q[3];
cx q[1],q[3];
ry(-1.4775150192177413) q[3];
ry(0.021513187265548517) q[5];
cx q[3],q[5];
ry(2.0156559609500326) q[3];
ry(-2.650248656209598) q[5];
cx q[3],q[5];
ry(-2.766416078805252) q[5];
ry(-0.05999329393578718) q[7];
cx q[5],q[7];
ry(-1.0760874942185088) q[5];
ry(-2.059014314453881) q[7];
cx q[5],q[7];
ry(-0.7494320237763937) q[0];
ry(1.2541091867026672) q[3];
cx q[0],q[3];
ry(-2.3275755627073407) q[0];
ry(-2.159234564288578) q[3];
cx q[0],q[3];
ry(-2.42421773469949) q[1];
ry(-0.6846395064804822) q[2];
cx q[1],q[2];
ry(0.8252626422816629) q[1];
ry(-2.957927593125554) q[2];
cx q[1],q[2];
ry(0.8794436125636215) q[2];
ry(2.28235364072758) q[5];
cx q[2],q[5];
ry(-0.0887948881761516) q[2];
ry(1.6219074108841856) q[5];
cx q[2],q[5];
ry(1.885050149683031) q[3];
ry(1.9375090303692606) q[4];
cx q[3],q[4];
ry(0.13205184887984966) q[3];
ry(-2.5628465475949853) q[4];
cx q[3],q[4];
ry(-1.1371675752461647) q[4];
ry(2.5510751226573896) q[7];
cx q[4],q[7];
ry(-3.122189134337936) q[4];
ry(0.4067762760900074) q[7];
cx q[4],q[7];
ry(0.018068848414106428) q[5];
ry(-2.784454472608401) q[6];
cx q[5],q[6];
ry(0.5872599525442542) q[5];
ry(-0.1247632734962254) q[6];
cx q[5],q[6];
ry(-0.1515279540902439) q[0];
ry(1.5684558354315534) q[1];
cx q[0],q[1];
ry(1.4748491575850402) q[0];
ry(3.0407759022042833) q[1];
cx q[0],q[1];
ry(0.9854738753748543) q[2];
ry(1.2569080547696077) q[3];
cx q[2],q[3];
ry(-2.878881975960552) q[2];
ry(2.4730509721483975) q[3];
cx q[2],q[3];
ry(2.9590147300725844) q[4];
ry(1.7774636994218902) q[5];
cx q[4],q[5];
ry(1.999189096205777) q[4];
ry(2.298616512920061) q[5];
cx q[4],q[5];
ry(-1.232410360639939) q[6];
ry(-1.8838135989860865) q[7];
cx q[6],q[7];
ry(1.7585407774427544) q[6];
ry(-2.8648735417654883) q[7];
cx q[6],q[7];
ry(-2.348752760579867) q[0];
ry(1.5729570999262361) q[2];
cx q[0],q[2];
ry(-2.543574330066802) q[0];
ry(-0.029670392327260358) q[2];
cx q[0],q[2];
ry(-1.0512240119181921) q[2];
ry(-2.803020012325737) q[4];
cx q[2],q[4];
ry(-1.3577352342463804) q[2];
ry(2.4807517686182514) q[4];
cx q[2],q[4];
ry(-2.7742527927067613) q[4];
ry(2.680341499601289) q[6];
cx q[4],q[6];
ry(1.4791566689792024) q[4];
ry(2.7514519999209504) q[6];
cx q[4],q[6];
ry(1.1538550612500535) q[1];
ry(-0.3179071995516853) q[3];
cx q[1],q[3];
ry(2.0642068073123485) q[1];
ry(1.637769417900903) q[3];
cx q[1],q[3];
ry(-0.8832372671493358) q[3];
ry(-2.5153002191896943) q[5];
cx q[3],q[5];
ry(-1.1550500633330048) q[3];
ry(0.4628495711396216) q[5];
cx q[3],q[5];
ry(0.3011071711686766) q[5];
ry(-0.9585914310485673) q[7];
cx q[5],q[7];
ry(1.3362554943863287) q[5];
ry(0.5556751097474084) q[7];
cx q[5],q[7];
ry(-2.70954828417689) q[0];
ry(0.39547043202749066) q[3];
cx q[0],q[3];
ry(-1.4668424649970637) q[0];
ry(-2.216695058117011) q[3];
cx q[0],q[3];
ry(1.192538650926859) q[1];
ry(-2.7131003536253577) q[2];
cx q[1],q[2];
ry(-0.5134066272472486) q[1];
ry(1.3438400736212879) q[2];
cx q[1],q[2];
ry(2.794936521142448) q[2];
ry(-1.8678649923208512) q[5];
cx q[2],q[5];
ry(-1.440654816925216) q[2];
ry(0.5568559866425531) q[5];
cx q[2],q[5];
ry(-0.7864817788214903) q[3];
ry(2.073588734571492) q[4];
cx q[3],q[4];
ry(2.8135632451653456) q[3];
ry(0.337267953212003) q[4];
cx q[3],q[4];
ry(0.982928467956551) q[4];
ry(-2.7602685263138604) q[7];
cx q[4],q[7];
ry(-2.89217269823221) q[4];
ry(-2.5452659792925596) q[7];
cx q[4],q[7];
ry(0.8965594535971926) q[5];
ry(2.5728937327497454) q[6];
cx q[5],q[6];
ry(-0.728091700715012) q[5];
ry(-0.21019721077443346) q[6];
cx q[5],q[6];
ry(-0.7036159711997534) q[0];
ry(1.4212124523023728) q[1];
ry(2.481459440417457) q[2];
ry(-1.2542208957203964) q[3];
ry(-1.7884275602362074) q[4];
ry(-0.6908557997112998) q[5];
ry(0.10566400092633099) q[6];
ry(-2.0428941604506816) q[7];