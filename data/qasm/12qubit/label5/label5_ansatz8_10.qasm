OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.6435383669069523) q[0];
ry(-0.6208283799050766) q[1];
cx q[0],q[1];
ry(-0.8466470148167163) q[0];
ry(0.4373336629085713) q[1];
cx q[0],q[1];
ry(-0.00903150266499253) q[2];
ry(-2.0803827066110654) q[3];
cx q[2],q[3];
ry(-1.176328775467748) q[2];
ry(1.956212061663078) q[3];
cx q[2],q[3];
ry(-0.3081088741768286) q[4];
ry(-3.0207024501982307) q[5];
cx q[4],q[5];
ry(-1.3076645863900724) q[4];
ry(-0.5308117244240623) q[5];
cx q[4],q[5];
ry(0.34388547855857166) q[6];
ry(2.173475696028794) q[7];
cx q[6],q[7];
ry(2.2795853320862083) q[6];
ry(0.6060745497634249) q[7];
cx q[6],q[7];
ry(0.677028649126771) q[8];
ry(2.6906137775140673) q[9];
cx q[8],q[9];
ry(-0.027917055774194562) q[8];
ry(0.5935523951415984) q[9];
cx q[8],q[9];
ry(1.4783865412445236) q[10];
ry(1.8277090584308846) q[11];
cx q[10],q[11];
ry(2.42624863358764) q[10];
ry(-2.02239968261079) q[11];
cx q[10],q[11];
ry(-2.993010374644551) q[0];
ry(-3.1350274949024297) q[2];
cx q[0],q[2];
ry(-1.5625484407184116) q[0];
ry(-2.878301992607707) q[2];
cx q[0],q[2];
ry(1.5873170465892112) q[2];
ry(-1.0792137429724453) q[4];
cx q[2],q[4];
ry(-1.003221920344053) q[2];
ry(1.2007758533951531) q[4];
cx q[2],q[4];
ry(3.1414699305840714) q[4];
ry(-0.6433760161203145) q[6];
cx q[4],q[6];
ry(-2.3241981104909066) q[4];
ry(-2.8085950581302557) q[6];
cx q[4],q[6];
ry(0.8216824697368388) q[6];
ry(-2.750197097590535) q[8];
cx q[6],q[8];
ry(0.6912492744824753) q[6];
ry(2.278403810699951) q[8];
cx q[6],q[8];
ry(-2.506583389187732) q[8];
ry(2.6921580747030633) q[10];
cx q[8],q[10];
ry(-0.1024054685361671) q[8];
ry(-1.2012822270350991) q[10];
cx q[8],q[10];
ry(-1.413782647177996) q[1];
ry(1.518543644968255) q[3];
cx q[1],q[3];
ry(-1.1141116671344806) q[1];
ry(2.208918287084407) q[3];
cx q[1],q[3];
ry(-1.7875992569077515) q[3];
ry(2.1770153394061458) q[5];
cx q[3],q[5];
ry(-2.5017565201176355) q[3];
ry(1.8817652881241547) q[5];
cx q[3],q[5];
ry(-1.4854131274226283) q[5];
ry(1.535354493011523) q[7];
cx q[5],q[7];
ry(1.9533620447444608) q[5];
ry(-1.875815785658311) q[7];
cx q[5],q[7];
ry(0.6546564520378535) q[7];
ry(-2.8416015401505326) q[9];
cx q[7],q[9];
ry(-1.6222089820320285) q[7];
ry(-2.2120828413741442) q[9];
cx q[7],q[9];
ry(1.9605711200040101) q[9];
ry(-0.08722637885760953) q[11];
cx q[9],q[11];
ry(-2.0644627304428527) q[9];
ry(-0.3355383361909947) q[11];
cx q[9],q[11];
ry(-0.3552193696918548) q[0];
ry(0.6102730451168243) q[1];
cx q[0],q[1];
ry(-0.5391880364408239) q[0];
ry(2.9123425092831323) q[1];
cx q[0],q[1];
ry(-2.199551280532124) q[2];
ry(-0.021262938566179557) q[3];
cx q[2],q[3];
ry(1.5194225593373403) q[2];
ry(2.398847533985042) q[3];
cx q[2],q[3];
ry(0.6503298200057052) q[4];
ry(2.5765487536029745) q[5];
cx q[4],q[5];
ry(1.5794344433081395) q[4];
ry(2.821556662482558) q[5];
cx q[4],q[5];
ry(-2.7152308358293697) q[6];
ry(-2.396532149306641) q[7];
cx q[6],q[7];
ry(-1.4321292093907616) q[6];
ry(-0.41229071569223236) q[7];
cx q[6],q[7];
ry(-0.9948435088756638) q[8];
ry(-1.502050248684384) q[9];
cx q[8],q[9];
ry(1.2757221766617075) q[8];
ry(1.722901003378209) q[9];
cx q[8],q[9];
ry(2.6088528254953283) q[10];
ry(0.35740435803319165) q[11];
cx q[10],q[11];
ry(2.4513611408092757) q[10];
ry(0.45894125954449194) q[11];
cx q[10],q[11];
ry(2.9882198687430837) q[0];
ry(-1.3717947951614358) q[2];
cx q[0],q[2];
ry(-0.8547238655285758) q[0];
ry(-1.5055138363629845) q[2];
cx q[0],q[2];
ry(-1.2355759155354162) q[2];
ry(1.0127813323177437) q[4];
cx q[2],q[4];
ry(-2.130266176846704) q[2];
ry(2.6250225675737715) q[4];
cx q[2],q[4];
ry(2.547199853851641) q[4];
ry(-1.478754077464953) q[6];
cx q[4],q[6];
ry(0.8910154312993861) q[4];
ry(-1.059709474159393) q[6];
cx q[4],q[6];
ry(0.5103656709241327) q[6];
ry(2.4787732452608906) q[8];
cx q[6],q[8];
ry(-2.3281338025833134) q[6];
ry(-0.9272672804361819) q[8];
cx q[6],q[8];
ry(-1.907884424463625) q[8];
ry(-2.763734752842862) q[10];
cx q[8],q[10];
ry(-2.902126467386697) q[8];
ry(-1.1510627979832473) q[10];
cx q[8],q[10];
ry(-1.2740035487034973) q[1];
ry(1.2921807610247376) q[3];
cx q[1],q[3];
ry(2.9345840928744553) q[1];
ry(-2.402488485777125) q[3];
cx q[1],q[3];
ry(1.8916166705516853) q[3];
ry(1.5922344914422542) q[5];
cx q[3],q[5];
ry(-2.260586107209598) q[3];
ry(0.5418779576262561) q[5];
cx q[3],q[5];
ry(1.1499708665293427) q[5];
ry(-2.560793723106195) q[7];
cx q[5],q[7];
ry(2.4079500852758544) q[5];
ry(-0.35201060667009537) q[7];
cx q[5],q[7];
ry(1.0284977143286156) q[7];
ry(-0.24724890660098797) q[9];
cx q[7],q[9];
ry(0.6950018091670579) q[7];
ry(-0.9966438895231741) q[9];
cx q[7],q[9];
ry(0.357758192357702) q[9];
ry(0.14520168564171165) q[11];
cx q[9],q[11];
ry(-2.7015310313874488) q[9];
ry(-1.7404006693791851) q[11];
cx q[9],q[11];
ry(-0.3023882587801836) q[0];
ry(2.7264358957116492) q[1];
cx q[0],q[1];
ry(-1.8514701122334136) q[0];
ry(2.0858938639506484) q[1];
cx q[0],q[1];
ry(-1.0118543035165282) q[2];
ry(1.014409439190274) q[3];
cx q[2],q[3];
ry(-0.4917831852938182) q[2];
ry(0.628714414802732) q[3];
cx q[2],q[3];
ry(1.340797528748196) q[4];
ry(3.1346790216994496) q[5];
cx q[4],q[5];
ry(-2.249464518734844) q[4];
ry(1.3900003179620155) q[5];
cx q[4],q[5];
ry(-1.03408088349516) q[6];
ry(-1.2159889600788798) q[7];
cx q[6],q[7];
ry(-0.7323575669343924) q[6];
ry(-0.8051662587916447) q[7];
cx q[6],q[7];
ry(1.1518894476802146) q[8];
ry(-0.019667249583267078) q[9];
cx q[8],q[9];
ry(-2.7979989231524818) q[8];
ry(1.9651110993280858) q[9];
cx q[8],q[9];
ry(-0.8477698313362977) q[10];
ry(0.15705962076941163) q[11];
cx q[10],q[11];
ry(2.1812774212037) q[10];
ry(-2.055435248816358) q[11];
cx q[10],q[11];
ry(1.5729953052090428) q[0];
ry(1.6255977305388107) q[2];
cx q[0],q[2];
ry(2.0987996518745744) q[0];
ry(2.8238618950149514) q[2];
cx q[0],q[2];
ry(-1.173909487748417) q[2];
ry(1.7086872684005268) q[4];
cx q[2],q[4];
ry(-2.6869812860569393) q[2];
ry(0.2836585345704073) q[4];
cx q[2],q[4];
ry(2.759805209521521) q[4];
ry(-2.3888903896568614) q[6];
cx q[4],q[6];
ry(0.8122467724957971) q[4];
ry(-2.079833468968191) q[6];
cx q[4],q[6];
ry(1.882288365834146) q[6];
ry(2.676210890542858) q[8];
cx q[6],q[8];
ry(2.141016178944668) q[6];
ry(2.7003649770438054) q[8];
cx q[6],q[8];
ry(-2.8464849304885895) q[8];
ry(-2.3046504768603104) q[10];
cx q[8],q[10];
ry(2.5891892286097806) q[8];
ry(2.0838493838886767) q[10];
cx q[8],q[10];
ry(1.130114638087158) q[1];
ry(0.8413889760379182) q[3];
cx q[1],q[3];
ry(2.3292736508998746) q[1];
ry(-1.4000484039946044) q[3];
cx q[1],q[3];
ry(-2.2608450132504805) q[3];
ry(2.8365303364182717) q[5];
cx q[3],q[5];
ry(-0.38963038645111187) q[3];
ry(-1.8760582708692637) q[5];
cx q[3],q[5];
ry(-2.007345802154582) q[5];
ry(2.0802305469086027) q[7];
cx q[5],q[7];
ry(1.7063935647498407) q[5];
ry(-2.2839640546761273) q[7];
cx q[5],q[7];
ry(-2.9077480550091828) q[7];
ry(-1.7220722191206637) q[9];
cx q[7],q[9];
ry(1.9397340270311636) q[7];
ry(-0.5972544741183388) q[9];
cx q[7],q[9];
ry(-0.3984317474615072) q[9];
ry(-2.7524494734004223) q[11];
cx q[9],q[11];
ry(-0.5423423156189671) q[9];
ry(2.3869143873967107) q[11];
cx q[9],q[11];
ry(-3.0819054847757297) q[0];
ry(2.5146119516449326) q[1];
cx q[0],q[1];
ry(0.7987045606180043) q[0];
ry(-1.5139158293458728) q[1];
cx q[0],q[1];
ry(2.375220399129775) q[2];
ry(-1.1377149805538034) q[3];
cx q[2],q[3];
ry(2.4069343429664904) q[2];
ry(-2.973305096549606) q[3];
cx q[2],q[3];
ry(-0.021564782397754223) q[4];
ry(-0.9483546087365973) q[5];
cx q[4],q[5];
ry(-1.99046997077786) q[4];
ry(-2.2922207762472246) q[5];
cx q[4],q[5];
ry(-3.0632792511289497) q[6];
ry(-1.5765094604390395) q[7];
cx q[6],q[7];
ry(2.4779011977505374) q[6];
ry(1.918367517766531) q[7];
cx q[6],q[7];
ry(2.292985398830435) q[8];
ry(-1.4889443861638598) q[9];
cx q[8],q[9];
ry(-1.3390093606752442) q[8];
ry(-2.1967394271341285) q[9];
cx q[8],q[9];
ry(2.2696737906560496) q[10];
ry(1.109851847540515) q[11];
cx q[10],q[11];
ry(2.555018168936016) q[10];
ry(2.706011084758834) q[11];
cx q[10],q[11];
ry(-1.8647581447872286) q[0];
ry(-1.0231354838977342) q[2];
cx q[0],q[2];
ry(-1.8439029967004255) q[0];
ry(0.46395417066384326) q[2];
cx q[0],q[2];
ry(2.144080962396627) q[2];
ry(0.8985105548443728) q[4];
cx q[2],q[4];
ry(-1.9720710291643049) q[2];
ry(-1.8854890780700488) q[4];
cx q[2],q[4];
ry(1.3469066654465722) q[4];
ry(1.4243271870011567) q[6];
cx q[4],q[6];
ry(-2.272187595228144) q[4];
ry(1.5904844654445303) q[6];
cx q[4],q[6];
ry(2.9657405450022707) q[6];
ry(0.9096760965405304) q[8];
cx q[6],q[8];
ry(-1.0575547222667936) q[6];
ry(0.562746778796093) q[8];
cx q[6],q[8];
ry(-1.840384763154294) q[8];
ry(0.48908896317001077) q[10];
cx q[8],q[10];
ry(2.823531357521891) q[8];
ry(2.736850288965183) q[10];
cx q[8],q[10];
ry(1.5902840867773795) q[1];
ry(0.1037321416124799) q[3];
cx q[1],q[3];
ry(0.12692798244586712) q[1];
ry(1.071212975710455) q[3];
cx q[1],q[3];
ry(-2.311416309056491) q[3];
ry(-1.4186388793620206) q[5];
cx q[3],q[5];
ry(1.2514179880240093) q[3];
ry(0.18539245798052484) q[5];
cx q[3],q[5];
ry(-0.6631134365947977) q[5];
ry(1.874599407957449) q[7];
cx q[5],q[7];
ry(3.0327860164275635) q[5];
ry(-2.6554624439477044) q[7];
cx q[5],q[7];
ry(0.29344469831118136) q[7];
ry(0.349321608445564) q[9];
cx q[7],q[9];
ry(-1.2404428636560825) q[7];
ry(2.008166267817833) q[9];
cx q[7],q[9];
ry(2.2909863867930547) q[9];
ry(-1.6080625911187392) q[11];
cx q[9],q[11];
ry(-2.0149916753608212) q[9];
ry(0.6450910015619981) q[11];
cx q[9],q[11];
ry(1.157409286132371) q[0];
ry(-2.539215125093438) q[1];
cx q[0],q[1];
ry(1.5902895948844726) q[0];
ry(2.8791483155029147) q[1];
cx q[0],q[1];
ry(0.45952517003804516) q[2];
ry(0.029214610076241685) q[3];
cx q[2],q[3];
ry(0.7939588251715515) q[2];
ry(-0.48354815098813564) q[3];
cx q[2],q[3];
ry(-1.2511767178012745) q[4];
ry(-1.7946563222133962) q[5];
cx q[4],q[5];
ry(2.505797915338885) q[4];
ry(-1.5103872399942257) q[5];
cx q[4],q[5];
ry(0.7443014871852505) q[6];
ry(0.21249647238795252) q[7];
cx q[6],q[7];
ry(-2.630942808126394) q[6];
ry(2.5227170691113763) q[7];
cx q[6],q[7];
ry(0.006337720973937699) q[8];
ry(-2.119389814082451) q[9];
cx q[8],q[9];
ry(1.3122612314588329) q[8];
ry(0.1325806495960651) q[9];
cx q[8],q[9];
ry(-0.18410814165149514) q[10];
ry(2.211553809472308) q[11];
cx q[10],q[11];
ry(-2.727166566617203) q[10];
ry(-0.5149788929099728) q[11];
cx q[10],q[11];
ry(-0.24438154686677188) q[0];
ry(2.4402620372131825) q[2];
cx q[0],q[2];
ry(-2.4853124824050075) q[0];
ry(2.6141390873711505) q[2];
cx q[0],q[2];
ry(-2.552878221034302) q[2];
ry(0.8546242306919515) q[4];
cx q[2],q[4];
ry(0.9968051134080025) q[2];
ry(2.4613780717708273) q[4];
cx q[2],q[4];
ry(2.686728288304553) q[4];
ry(2.11340495282818) q[6];
cx q[4],q[6];
ry(1.5840725839187404) q[4];
ry(-0.545721302845739) q[6];
cx q[4],q[6];
ry(-0.10118055809770712) q[6];
ry(2.104421296445291) q[8];
cx q[6],q[8];
ry(-0.6593754137168872) q[6];
ry(-0.72206492771099) q[8];
cx q[6],q[8];
ry(0.7950650380500859) q[8];
ry(-0.9595082986621453) q[10];
cx q[8],q[10];
ry(2.141124613937528) q[8];
ry(1.9355865521974884) q[10];
cx q[8],q[10];
ry(0.5512673807201286) q[1];
ry(-2.3409829130446993) q[3];
cx q[1],q[3];
ry(-1.2260219596503703) q[1];
ry(0.7250526913246045) q[3];
cx q[1],q[3];
ry(1.1965932792643914) q[3];
ry(-1.9939107803563973) q[5];
cx q[3],q[5];
ry(-1.3285647760816144) q[3];
ry(0.513251663610431) q[5];
cx q[3],q[5];
ry(-2.184033608701383) q[5];
ry(0.22476128193748776) q[7];
cx q[5],q[7];
ry(2.9964556560071034) q[5];
ry(-0.9301184270801208) q[7];
cx q[5],q[7];
ry(-1.1482164035683267) q[7];
ry(-2.9553955242991043) q[9];
cx q[7],q[9];
ry(1.916982943171793) q[7];
ry(-2.056237136702549) q[9];
cx q[7],q[9];
ry(-2.8670953543965503) q[9];
ry(0.08531156056926514) q[11];
cx q[9],q[11];
ry(-1.1071609369340687) q[9];
ry(1.3833001990713925) q[11];
cx q[9],q[11];
ry(0.21986682600826707) q[0];
ry(1.0725742430867606) q[1];
cx q[0],q[1];
ry(1.3043148916028664) q[0];
ry(-1.864981077884738) q[1];
cx q[0],q[1];
ry(2.709173056014487) q[2];
ry(1.5957239603431375) q[3];
cx q[2],q[3];
ry(-2.901354910526318) q[2];
ry(-2.2543166321124217) q[3];
cx q[2],q[3];
ry(2.127200017387295) q[4];
ry(1.172728090543525) q[5];
cx q[4],q[5];
ry(-1.8645575688748506) q[4];
ry(-2.874137157473658) q[5];
cx q[4],q[5];
ry(0.6746064750405093) q[6];
ry(-0.5053090846180925) q[7];
cx q[6],q[7];
ry(-1.1854617151925713) q[6];
ry(0.32560250975605864) q[7];
cx q[6],q[7];
ry(2.2468247587350634) q[8];
ry(-0.7481982573538888) q[9];
cx q[8],q[9];
ry(-2.7341541612344424) q[8];
ry(-0.7493016960421648) q[9];
cx q[8],q[9];
ry(-3.057642075745197) q[10];
ry(-2.3831205332756533) q[11];
cx q[10],q[11];
ry(0.5990655580655057) q[10];
ry(-2.0647015745660444) q[11];
cx q[10],q[11];
ry(-1.8247653305077653) q[0];
ry(-2.206281272457866) q[2];
cx q[0],q[2];
ry(-2.2311203984110115) q[0];
ry(0.3581771765821661) q[2];
cx q[0],q[2];
ry(2.0909542015407174) q[2];
ry(-1.8731664641910033) q[4];
cx q[2],q[4];
ry(0.9130448612420063) q[2];
ry(-1.324440543824796) q[4];
cx q[2],q[4];
ry(1.8997708805716007) q[4];
ry(-0.9711013371472939) q[6];
cx q[4],q[6];
ry(-1.5509385900115102) q[4];
ry(1.3512948502125264) q[6];
cx q[4],q[6];
ry(2.084570251879505) q[6];
ry(2.653998156488648) q[8];
cx q[6],q[8];
ry(1.3423869135453612) q[6];
ry(2.0011463357540427) q[8];
cx q[6],q[8];
ry(1.5346750188959866) q[8];
ry(1.6082428304463214) q[10];
cx q[8],q[10];
ry(0.3408424860728073) q[8];
ry(0.8159804015535981) q[10];
cx q[8],q[10];
ry(1.7856211735510072) q[1];
ry(2.0339808005596804) q[3];
cx q[1],q[3];
ry(2.3656309663179194) q[1];
ry(-3.135358329831046) q[3];
cx q[1],q[3];
ry(-0.7111985510080848) q[3];
ry(-0.718764468432646) q[5];
cx q[3],q[5];
ry(-1.0857383181362028) q[3];
ry(2.9837206128621734) q[5];
cx q[3],q[5];
ry(1.802895796109536) q[5];
ry(-0.04578971234255835) q[7];
cx q[5],q[7];
ry(-0.17427327729448963) q[5];
ry(2.4242178813974937) q[7];
cx q[5],q[7];
ry(-1.52605098477276) q[7];
ry(-2.3117083448141114) q[9];
cx q[7],q[9];
ry(0.18101373340929253) q[7];
ry(-1.8546393126224665) q[9];
cx q[7],q[9];
ry(0.37932361997700426) q[9];
ry(-1.4915933288083254) q[11];
cx q[9],q[11];
ry(0.5026166832740422) q[9];
ry(-2.906981205901256) q[11];
cx q[9],q[11];
ry(1.0507280240336412) q[0];
ry(2.3332376508389023) q[1];
cx q[0],q[1];
ry(-1.0572116354263787) q[0];
ry(-2.2580970184677276) q[1];
cx q[0],q[1];
ry(0.5725341787521047) q[2];
ry(-1.3031113538438288) q[3];
cx q[2],q[3];
ry(-3.0532526809718634) q[2];
ry(-1.9680851331196938) q[3];
cx q[2],q[3];
ry(1.4463297424372241) q[4];
ry(-2.112821202630703) q[5];
cx q[4],q[5];
ry(-0.7365422948417636) q[4];
ry(-2.2806917199597425) q[5];
cx q[4],q[5];
ry(-1.9539995952055176) q[6];
ry(-1.1276609637397224) q[7];
cx q[6],q[7];
ry(-2.2029359098981853) q[6];
ry(1.1165068684705135) q[7];
cx q[6],q[7];
ry(2.6035573415531905) q[8];
ry(1.3720118733316833) q[9];
cx q[8],q[9];
ry(0.8486930073694869) q[8];
ry(-3.0872391186294146) q[9];
cx q[8],q[9];
ry(2.6631236970477805) q[10];
ry(-1.0937356134010212) q[11];
cx q[10],q[11];
ry(-1.8830309395236657) q[10];
ry(1.47542447717209) q[11];
cx q[10],q[11];
ry(-2.797865583710078) q[0];
ry(0.7316190683066734) q[2];
cx q[0],q[2];
ry(0.8256524776670733) q[0];
ry(2.888469710053834) q[2];
cx q[0],q[2];
ry(0.6452293120869355) q[2];
ry(1.8934941788865611) q[4];
cx q[2],q[4];
ry(0.2631470843204308) q[2];
ry(-2.4236383540621294) q[4];
cx q[2],q[4];
ry(-0.2702687993264279) q[4];
ry(1.2844462932693654) q[6];
cx q[4],q[6];
ry(2.06007712821031) q[4];
ry(-2.493804756610494) q[6];
cx q[4],q[6];
ry(0.6486741070572083) q[6];
ry(-1.7123578923391556) q[8];
cx q[6],q[8];
ry(2.7432831402032845) q[6];
ry(-2.6666063030064286) q[8];
cx q[6],q[8];
ry(0.8998460934758733) q[8];
ry(2.6973143991791027) q[10];
cx q[8],q[10];
ry(1.790737861260431) q[8];
ry(2.7392835838383385) q[10];
cx q[8],q[10];
ry(0.6535322036456286) q[1];
ry(0.0496272804205602) q[3];
cx q[1],q[3];
ry(1.2268738052726311) q[1];
ry(-1.8632409215729835) q[3];
cx q[1],q[3];
ry(-0.22688452878878887) q[3];
ry(1.9162653453324705) q[5];
cx q[3],q[5];
ry(-1.4764325627719304) q[3];
ry(1.93031993829835) q[5];
cx q[3],q[5];
ry(-2.995628116344276) q[5];
ry(2.035716919080378) q[7];
cx q[5],q[7];
ry(2.4701466970841564) q[5];
ry(2.7265221065888134) q[7];
cx q[5],q[7];
ry(0.978333789698465) q[7];
ry(2.284854389353822) q[9];
cx q[7],q[9];
ry(2.4897975522438593) q[7];
ry(-0.9819454718386084) q[9];
cx q[7],q[9];
ry(0.3884005979817299) q[9];
ry(-2.011053130168947) q[11];
cx q[9],q[11];
ry(2.5331520132022654) q[9];
ry(-1.4333761315953835) q[11];
cx q[9],q[11];
ry(0.9433329804996538) q[0];
ry(-2.152235438821534) q[1];
cx q[0],q[1];
ry(-1.047953560427299) q[0];
ry(0.8440621620743514) q[1];
cx q[0],q[1];
ry(-0.5205927490961804) q[2];
ry(-2.6578475899637914) q[3];
cx q[2],q[3];
ry(2.9902391820777714) q[2];
ry(-1.3321422598297417) q[3];
cx q[2],q[3];
ry(1.5860366099620447) q[4];
ry(3.0114838975379654) q[5];
cx q[4],q[5];
ry(2.7206861237520323) q[4];
ry(-0.7344042195391157) q[5];
cx q[4],q[5];
ry(0.5413845218482969) q[6];
ry(-0.077655555380849) q[7];
cx q[6],q[7];
ry(-1.5167785262728) q[6];
ry(2.3995287592854875) q[7];
cx q[6],q[7];
ry(-0.7024899593315227) q[8];
ry(-0.2801622053079855) q[9];
cx q[8],q[9];
ry(-1.1653524341546135) q[8];
ry(-1.7902179964694893) q[9];
cx q[8],q[9];
ry(2.4886365123496983) q[10];
ry(0.7036113972888378) q[11];
cx q[10],q[11];
ry(-0.37635647887228973) q[10];
ry(0.9126841913009816) q[11];
cx q[10],q[11];
ry(3.064841867686845) q[0];
ry(-1.8007784075170714) q[2];
cx q[0],q[2];
ry(-2.5346613762962287) q[0];
ry(0.6154492338122326) q[2];
cx q[0],q[2];
ry(1.9284527437006638) q[2];
ry(1.0927765557287423) q[4];
cx q[2],q[4];
ry(-1.3600485434976104) q[2];
ry(1.5151080936426897) q[4];
cx q[2],q[4];
ry(2.5253205081178804) q[4];
ry(-1.802496797131762) q[6];
cx q[4],q[6];
ry(-1.3720294778097353) q[4];
ry(3.033965564233361) q[6];
cx q[4],q[6];
ry(-1.31008171440145) q[6];
ry(-1.9748094415643414) q[8];
cx q[6],q[8];
ry(-1.911157374658333) q[6];
ry(1.4164083338091586) q[8];
cx q[6],q[8];
ry(-0.5796225876962977) q[8];
ry(0.0717246989191791) q[10];
cx q[8],q[10];
ry(-1.5623552951021509) q[8];
ry(-1.7528945413447614) q[10];
cx q[8],q[10];
ry(-1.2314265036943803) q[1];
ry(1.470262626573669) q[3];
cx q[1],q[3];
ry(2.3255791962975025) q[1];
ry(0.923739599336141) q[3];
cx q[1],q[3];
ry(-0.5082785606457119) q[3];
ry(-2.3504750918664277) q[5];
cx q[3],q[5];
ry(-2.8126377310044437) q[3];
ry(0.045307397336755706) q[5];
cx q[3],q[5];
ry(-2.792567406749095) q[5];
ry(2.764413828691932) q[7];
cx q[5],q[7];
ry(2.5636056436623718) q[5];
ry(-2.952433821758906) q[7];
cx q[5],q[7];
ry(-3.037035180175097) q[7];
ry(-2.8939059102856617) q[9];
cx q[7],q[9];
ry(2.840487387377203) q[7];
ry(-1.3794792283336088) q[9];
cx q[7],q[9];
ry(-0.10080658205629783) q[9];
ry(-1.156386279609217) q[11];
cx q[9],q[11];
ry(-2.373589766865031) q[9];
ry(0.43401905618553815) q[11];
cx q[9],q[11];
ry(-2.8576041821821114) q[0];
ry(-0.06951891496432784) q[1];
cx q[0],q[1];
ry(1.2281432571971824) q[0];
ry(0.484824929457667) q[1];
cx q[0],q[1];
ry(-1.0481041686828736) q[2];
ry(-1.0995914611842181) q[3];
cx q[2],q[3];
ry(-0.8528979161104031) q[2];
ry(-0.6333378975341155) q[3];
cx q[2],q[3];
ry(0.3485336390821033) q[4];
ry(2.4877055359604263) q[5];
cx q[4],q[5];
ry(1.9996984553297565) q[4];
ry(1.956764811375483) q[5];
cx q[4],q[5];
ry(1.974226026291874) q[6];
ry(0.8984125888821647) q[7];
cx q[6],q[7];
ry(-0.24635033641821824) q[6];
ry(1.9645942081460708) q[7];
cx q[6],q[7];
ry(-1.351219445895364) q[8];
ry(-2.227358719755097) q[9];
cx q[8],q[9];
ry(1.1226934636429375) q[8];
ry(2.377862217726853) q[9];
cx q[8],q[9];
ry(-2.8384608371335367) q[10];
ry(1.6157564646688902) q[11];
cx q[10],q[11];
ry(-2.6417004334651564) q[10];
ry(-0.4580335533620046) q[11];
cx q[10],q[11];
ry(0.6478715476405279) q[0];
ry(0.8858478509379414) q[2];
cx q[0],q[2];
ry(0.33827753128330773) q[0];
ry(0.49058750004045754) q[2];
cx q[0],q[2];
ry(-2.7294301791124433) q[2];
ry(-1.0825913452253886) q[4];
cx q[2],q[4];
ry(1.0706556400613279) q[2];
ry(-0.28634587747074014) q[4];
cx q[2],q[4];
ry(-1.7283759114342214) q[4];
ry(-2.8354242739853897) q[6];
cx q[4],q[6];
ry(2.2244094160875036) q[4];
ry(-2.194119855015061) q[6];
cx q[4],q[6];
ry(-1.8463215810539495) q[6];
ry(2.257493587059605) q[8];
cx q[6],q[8];
ry(-1.259909057151459) q[6];
ry(0.8930326297069396) q[8];
cx q[6],q[8];
ry(-0.4072973117733909) q[8];
ry(-0.7225821529641424) q[10];
cx q[8],q[10];
ry(-1.529251849405017) q[8];
ry(1.754559775007925) q[10];
cx q[8],q[10];
ry(1.0193376440930937) q[1];
ry(-1.8336721093334316) q[3];
cx q[1],q[3];
ry(-1.2059141929542925) q[1];
ry(2.206344769263303) q[3];
cx q[1],q[3];
ry(-0.46950644709232753) q[3];
ry(0.9163747835769761) q[5];
cx q[3],q[5];
ry(2.7895098020180127) q[3];
ry(1.9319333406202457) q[5];
cx q[3],q[5];
ry(-0.20991510221181764) q[5];
ry(-1.335505367057489) q[7];
cx q[5],q[7];
ry(-1.5024487578053591) q[5];
ry(-0.13122232764006944) q[7];
cx q[5],q[7];
ry(-3.0982365407028154) q[7];
ry(-2.4232813296495825) q[9];
cx q[7],q[9];
ry(2.552480880994927) q[7];
ry(2.1295000924402174) q[9];
cx q[7],q[9];
ry(-1.4970356838042975) q[9];
ry(1.9089311417466046) q[11];
cx q[9],q[11];
ry(1.916271174409105) q[9];
ry(-0.7575922076985153) q[11];
cx q[9],q[11];
ry(2.414599006438997) q[0];
ry(2.7394516256525185) q[1];
cx q[0],q[1];
ry(-0.5948623302928634) q[0];
ry(2.163795737622797) q[1];
cx q[0],q[1];
ry(-2.7131380874908193) q[2];
ry(-2.056117533037769) q[3];
cx q[2],q[3];
ry(1.8867723244402725) q[2];
ry(-2.5345072587757493) q[3];
cx q[2],q[3];
ry(1.9105039883729544) q[4];
ry(-1.0007336506351399) q[5];
cx q[4],q[5];
ry(-2.4162496973053975) q[4];
ry(-0.7234472023932341) q[5];
cx q[4],q[5];
ry(-0.9214255784900088) q[6];
ry(0.35172687200887304) q[7];
cx q[6],q[7];
ry(1.610791555423268) q[6];
ry(-1.8227043467583675) q[7];
cx q[6],q[7];
ry(-2.5905745914185543) q[8];
ry(-2.3229171021676778) q[9];
cx q[8],q[9];
ry(0.38418563561463315) q[8];
ry(2.0032182216915038) q[9];
cx q[8],q[9];
ry(-1.8482850482887336) q[10];
ry(-1.1695385655212256) q[11];
cx q[10],q[11];
ry(2.9182163607188536) q[10];
ry(0.860079695616488) q[11];
cx q[10],q[11];
ry(0.05983225805786585) q[0];
ry(2.624151851588128) q[2];
cx q[0],q[2];
ry(0.8847846714346632) q[0];
ry(2.0657565537225606) q[2];
cx q[0],q[2];
ry(-2.8156377383021245) q[2];
ry(1.3043687414148923) q[4];
cx q[2],q[4];
ry(-0.7226538858123082) q[2];
ry(-2.4494386728806297) q[4];
cx q[2],q[4];
ry(1.3283849632940903) q[4];
ry(-1.8916344358081636) q[6];
cx q[4],q[6];
ry(-1.9414803716117928) q[4];
ry(-1.7165024138546903) q[6];
cx q[4],q[6];
ry(-0.42804587323596127) q[6];
ry(2.897904146804341) q[8];
cx q[6],q[8];
ry(0.3505188650167197) q[6];
ry(2.3318176728916793) q[8];
cx q[6],q[8];
ry(2.541966127549308) q[8];
ry(0.16980023926892504) q[10];
cx q[8],q[10];
ry(0.8452171168705851) q[8];
ry(-1.2538267337300708) q[10];
cx q[8],q[10];
ry(-1.331271435116494) q[1];
ry(3.087348290290467) q[3];
cx q[1],q[3];
ry(2.8683071169896706) q[1];
ry(-0.45086571853544033) q[3];
cx q[1],q[3];
ry(0.9789572534558565) q[3];
ry(2.7948508508427836) q[5];
cx q[3],q[5];
ry(0.44210855530536347) q[3];
ry(2.487101991542392) q[5];
cx q[3],q[5];
ry(2.6218532700411563) q[5];
ry(0.41639678021938753) q[7];
cx q[5],q[7];
ry(1.04395833887464) q[5];
ry(-0.6773063354241852) q[7];
cx q[5],q[7];
ry(-1.7963550435505768) q[7];
ry(1.4468848764980937) q[9];
cx q[7],q[9];
ry(-2.7032206690026643) q[7];
ry(3.0452215433720955) q[9];
cx q[7],q[9];
ry(-2.014791807468195) q[9];
ry(1.2407664266821383) q[11];
cx q[9],q[11];
ry(-1.3020340834943969) q[9];
ry(-2.421553510181213) q[11];
cx q[9],q[11];
ry(-1.450042284926858) q[0];
ry(-2.826781065383372) q[1];
cx q[0],q[1];
ry(-0.3674576211040091) q[0];
ry(-0.7636548467216988) q[1];
cx q[0],q[1];
ry(1.3313669648491704) q[2];
ry(-1.0406437430230715) q[3];
cx q[2],q[3];
ry(2.8134147399426648) q[2];
ry(0.8241442983322491) q[3];
cx q[2],q[3];
ry(0.46582645631322206) q[4];
ry(-2.0125012079387226) q[5];
cx q[4],q[5];
ry(-1.494421720053337) q[4];
ry(1.1662471439898976) q[5];
cx q[4],q[5];
ry(-1.4060120763447408) q[6];
ry(-2.872092242692985) q[7];
cx q[6],q[7];
ry(2.779528541927951) q[6];
ry(-1.5565215610746552) q[7];
cx q[6],q[7];
ry(-0.2958634596563883) q[8];
ry(-1.6851739464938418) q[9];
cx q[8],q[9];
ry(1.9250890278172446) q[8];
ry(-1.0776452235122767) q[9];
cx q[8],q[9];
ry(-1.2713662161188268) q[10];
ry(-0.5980320078773583) q[11];
cx q[10],q[11];
ry(0.24435422775874382) q[10];
ry(2.9505314059027064) q[11];
cx q[10],q[11];
ry(2.914443730250602) q[0];
ry(-1.6694230805213943) q[2];
cx q[0],q[2];
ry(2.8122547599254646) q[0];
ry(1.363874636042687) q[2];
cx q[0],q[2];
ry(-2.4448937592218276) q[2];
ry(0.6065579857975938) q[4];
cx q[2],q[4];
ry(2.496948186789576) q[2];
ry(0.7480607797333054) q[4];
cx q[2],q[4];
ry(2.9870681683093583) q[4];
ry(-2.378734877189244) q[6];
cx q[4],q[6];
ry(-1.5585596889338231) q[4];
ry(-2.741017382562861) q[6];
cx q[4],q[6];
ry(0.6054237025870022) q[6];
ry(-3.005000164547655) q[8];
cx q[6],q[8];
ry(-1.3763642781883307) q[6];
ry(1.5241990706892201) q[8];
cx q[6],q[8];
ry(0.3785488667864101) q[8];
ry(1.9662153964789466) q[10];
cx q[8],q[10];
ry(-1.7368938235417826) q[8];
ry(2.570179472200935) q[10];
cx q[8],q[10];
ry(-0.746870561621197) q[1];
ry(-0.9356265672136737) q[3];
cx q[1],q[3];
ry(-0.9808997574718927) q[1];
ry(0.655228497813245) q[3];
cx q[1],q[3];
ry(-2.484956663874458) q[3];
ry(1.2738862970338432) q[5];
cx q[3],q[5];
ry(-2.4778225150866957) q[3];
ry(-2.2838281006754113) q[5];
cx q[3],q[5];
ry(2.6992812560354937) q[5];
ry(0.20828630332939646) q[7];
cx q[5],q[7];
ry(-1.855508768422609) q[5];
ry(0.7794006625798646) q[7];
cx q[5],q[7];
ry(2.367008020285341) q[7];
ry(1.9390605831903898) q[9];
cx q[7],q[9];
ry(-0.20667938194969213) q[7];
ry(2.577568821221151) q[9];
cx q[7],q[9];
ry(-0.10764733663520364) q[9];
ry(1.3456013695298754) q[11];
cx q[9],q[11];
ry(1.4028581141529344) q[9];
ry(-0.18768096758327335) q[11];
cx q[9],q[11];
ry(-0.6777591010147486) q[0];
ry(2.627966711924241) q[1];
cx q[0],q[1];
ry(2.460269052229683) q[0];
ry(1.4147808495277452) q[1];
cx q[0],q[1];
ry(-2.074135257144258) q[2];
ry(-1.2227381624208205) q[3];
cx q[2],q[3];
ry(0.7713462939827354) q[2];
ry(-1.6250391898404732) q[3];
cx q[2],q[3];
ry(-2.700841887839521) q[4];
ry(0.5784215152621988) q[5];
cx q[4],q[5];
ry(-0.09787009650968613) q[4];
ry(2.3548503240930367) q[5];
cx q[4],q[5];
ry(-2.3748073300689856) q[6];
ry(2.3164570622264073) q[7];
cx q[6],q[7];
ry(0.8139226249134807) q[6];
ry(0.5533720949052666) q[7];
cx q[6],q[7];
ry(-0.4285241761439442) q[8];
ry(0.09283458593417748) q[9];
cx q[8],q[9];
ry(2.8907557710571092) q[8];
ry(1.824811776845824) q[9];
cx q[8],q[9];
ry(-3.1114187622443987) q[10];
ry(-3.047884133275709) q[11];
cx q[10],q[11];
ry(-2.755372273771602) q[10];
ry(0.6719674079558726) q[11];
cx q[10],q[11];
ry(-1.6488220805804898) q[0];
ry(0.10266359914094779) q[2];
cx q[0],q[2];
ry(0.7076790261958354) q[0];
ry(0.8177156394661953) q[2];
cx q[0],q[2];
ry(2.42877581934044) q[2];
ry(-2.8338712229252905) q[4];
cx q[2],q[4];
ry(1.2084772029574857) q[2];
ry(-0.652564426695429) q[4];
cx q[2],q[4];
ry(2.2980251749601486) q[4];
ry(-1.3288321132751433) q[6];
cx q[4],q[6];
ry(-0.3050925639596763) q[4];
ry(0.8888435442281235) q[6];
cx q[4],q[6];
ry(2.4466754598297475) q[6];
ry(2.869470184435057) q[8];
cx q[6],q[8];
ry(-1.3211601876440522) q[6];
ry(0.884901047080004) q[8];
cx q[6],q[8];
ry(-3.1233647726536913) q[8];
ry(0.4764074895462924) q[10];
cx q[8],q[10];
ry(2.2209238942480063) q[8];
ry(-1.3991212072196832) q[10];
cx q[8],q[10];
ry(-0.8918707505965031) q[1];
ry(1.8745082028163784) q[3];
cx q[1],q[3];
ry(0.33703296283193435) q[1];
ry(2.768773837125664) q[3];
cx q[1],q[3];
ry(-2.9811228756647363) q[3];
ry(-1.5190834471423036) q[5];
cx q[3],q[5];
ry(-2.9593759439141047) q[3];
ry(-2.6300314784930925) q[5];
cx q[3],q[5];
ry(-1.0402225732980979) q[5];
ry(2.9095726287822568) q[7];
cx q[5],q[7];
ry(-2.195200045793304) q[5];
ry(1.3753206510691567) q[7];
cx q[5],q[7];
ry(0.7811693432588358) q[7];
ry(2.2505763756891928) q[9];
cx q[7],q[9];
ry(1.8534748743108909) q[7];
ry(2.566865030657752) q[9];
cx q[7],q[9];
ry(-1.9630662646026333) q[9];
ry(-1.8799216696656453) q[11];
cx q[9],q[11];
ry(1.8055163899403535) q[9];
ry(-1.3485812941615558) q[11];
cx q[9],q[11];
ry(0.7235916359217898) q[0];
ry(-1.4066628814594142) q[1];
cx q[0],q[1];
ry(-1.5288660790020874) q[0];
ry(0.9271376464413317) q[1];
cx q[0],q[1];
ry(-1.8396467245968804) q[2];
ry(2.077995021643682) q[3];
cx q[2],q[3];
ry(-1.8021629525786165) q[2];
ry(1.1233736295164578) q[3];
cx q[2],q[3];
ry(-1.241075947519462) q[4];
ry(-1.7417841867480837) q[5];
cx q[4],q[5];
ry(-2.7096766032328214) q[4];
ry(-2.57876169000003) q[5];
cx q[4],q[5];
ry(-2.7828926449957936) q[6];
ry(0.07600639657971708) q[7];
cx q[6],q[7];
ry(-1.6402626015197272) q[6];
ry(1.6619942725425239) q[7];
cx q[6],q[7];
ry(0.5083623034544457) q[8];
ry(-2.7073140236807287) q[9];
cx q[8],q[9];
ry(-1.0053317021587729) q[8];
ry(2.1184293115068678) q[9];
cx q[8],q[9];
ry(-0.9397190927970092) q[10];
ry(-1.6229507629290278) q[11];
cx q[10],q[11];
ry(0.584791162264537) q[10];
ry(-1.1437356183032117) q[11];
cx q[10],q[11];
ry(2.0742170758938845) q[0];
ry(0.3362824211549882) q[2];
cx q[0],q[2];
ry(-2.532814487105354) q[0];
ry(-1.781792748279111) q[2];
cx q[0],q[2];
ry(-3.1278473307676022) q[2];
ry(1.9332778600866014) q[4];
cx q[2],q[4];
ry(0.8938418928672629) q[2];
ry(-0.7307405518087142) q[4];
cx q[2],q[4];
ry(-0.01844360576614198) q[4];
ry(0.7425963518206631) q[6];
cx q[4],q[6];
ry(-2.750923389350787) q[4];
ry(-0.46570477607294875) q[6];
cx q[4],q[6];
ry(2.5611613327593035) q[6];
ry(1.099304798908137) q[8];
cx q[6],q[8];
ry(2.0265367890825234) q[6];
ry(1.4363290724467768) q[8];
cx q[6],q[8];
ry(-1.3597153446611703) q[8];
ry(-0.5510779020511476) q[10];
cx q[8],q[10];
ry(-0.7475009029102012) q[8];
ry(1.710986469346519) q[10];
cx q[8],q[10];
ry(0.01120535692819681) q[1];
ry(-2.4547518569627744) q[3];
cx q[1],q[3];
ry(1.9399735622875138) q[1];
ry(2.9195668057321664) q[3];
cx q[1],q[3];
ry(-2.80926479381424) q[3];
ry(0.9117412561942592) q[5];
cx q[3],q[5];
ry(2.950753560045043) q[3];
ry(-1.8421521963771434) q[5];
cx q[3],q[5];
ry(-2.6401334417163778) q[5];
ry(-1.5025463444414422) q[7];
cx q[5],q[7];
ry(1.0768259248791947) q[5];
ry(-0.676250710078432) q[7];
cx q[5],q[7];
ry(2.6816248531540268) q[7];
ry(-3.011806279436468) q[9];
cx q[7],q[9];
ry(0.48357476002957117) q[7];
ry(-2.517614685818922) q[9];
cx q[7],q[9];
ry(1.8595148440881513) q[9];
ry(2.5799438084705826) q[11];
cx q[9],q[11];
ry(2.2494822972364075) q[9];
ry(0.15188351394754787) q[11];
cx q[9],q[11];
ry(-2.251169473927095) q[0];
ry(1.1829691946052918) q[1];
ry(-0.8912946018824313) q[2];
ry(1.9918109936134725) q[3];
ry(0.7990938221091959) q[4];
ry(-2.561112450527352) q[5];
ry(2.8184948738173983) q[6];
ry(0.6428883073573797) q[7];
ry(0.651953197044671) q[8];
ry(2.8165506577666646) q[9];
ry(1.9720610228798288) q[10];
ry(0.23074212647635697) q[11];