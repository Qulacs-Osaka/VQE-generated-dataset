OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.768464337594918) q[0];
ry(2.445387501558933) q[1];
cx q[0],q[1];
ry(-1.8418284639888352) q[0];
ry(1.9013095231944843) q[1];
cx q[0],q[1];
ry(2.437921321488683) q[2];
ry(-1.3103833559110023) q[3];
cx q[2],q[3];
ry(-1.5699622443346637) q[2];
ry(2.0084431319000853) q[3];
cx q[2],q[3];
ry(-0.7963371854006595) q[4];
ry(-0.0433982273698188) q[5];
cx q[4],q[5];
ry(-2.179765107759307) q[4];
ry(1.3319458105164073) q[5];
cx q[4],q[5];
ry(-1.8116549704462137) q[6];
ry(0.9559544349658964) q[7];
cx q[6],q[7];
ry(-1.1542305858741617) q[6];
ry(2.339825561732702) q[7];
cx q[6],q[7];
ry(-1.292948579235426) q[0];
ry(2.4589317235110344) q[2];
cx q[0],q[2];
ry(-0.6015242333810793) q[0];
ry(-0.7300074194631785) q[2];
cx q[0],q[2];
ry(-0.9644617517416387) q[2];
ry(0.25619046188595684) q[4];
cx q[2],q[4];
ry(3.069761462139928) q[2];
ry(1.316316941942374) q[4];
cx q[2],q[4];
ry(-2.236912140880777) q[4];
ry(-0.502701349235023) q[6];
cx q[4],q[6];
ry(-2.628966610838394) q[4];
ry(-1.2394847937734843) q[6];
cx q[4],q[6];
ry(-1.4684114163549937) q[1];
ry(-0.08854823209138907) q[3];
cx q[1],q[3];
ry(-1.0751778599656492) q[1];
ry(0.186219181998009) q[3];
cx q[1],q[3];
ry(0.6272616207343953) q[3];
ry(-0.34100726516681445) q[5];
cx q[3],q[5];
ry(1.0245913657909989) q[3];
ry(-0.402755855997745) q[5];
cx q[3],q[5];
ry(2.7515806037686925) q[5];
ry(2.8669711464264) q[7];
cx q[5],q[7];
ry(0.9041763995065497) q[5];
ry(2.9755615390222414) q[7];
cx q[5],q[7];
ry(0.3293000930764318) q[0];
ry(0.7541620626991521) q[3];
cx q[0],q[3];
ry(-1.1998221127140518) q[0];
ry(2.6764629718831423) q[3];
cx q[0],q[3];
ry(-0.16488215101365) q[1];
ry(-0.22146383329809982) q[2];
cx q[1],q[2];
ry(-0.979634105859881) q[1];
ry(0.4656505501583668) q[2];
cx q[1],q[2];
ry(-0.4038685108454213) q[2];
ry(-0.6281008524587967) q[5];
cx q[2],q[5];
ry(1.7773947520215914) q[2];
ry(-1.2077156709512442) q[5];
cx q[2],q[5];
ry(-2.7261548017821977) q[3];
ry(2.3610068339751513) q[4];
cx q[3],q[4];
ry(-0.3484895344287325) q[3];
ry(0.8713977961717607) q[4];
cx q[3],q[4];
ry(2.2068229091178146) q[4];
ry(-0.10278440913698707) q[7];
cx q[4],q[7];
ry(1.0326313838933847) q[4];
ry(2.0516251517888993) q[7];
cx q[4],q[7];
ry(0.0308078342621938) q[5];
ry(1.4833683945824059) q[6];
cx q[5],q[6];
ry(2.58305846678868) q[5];
ry(2.9144904090645287) q[6];
cx q[5],q[6];
ry(1.1299717841710677) q[0];
ry(-2.7521769541481853) q[1];
cx q[0],q[1];
ry(-1.440298024909786) q[0];
ry(-2.086644199862136) q[1];
cx q[0],q[1];
ry(1.770338188787064) q[2];
ry(-2.238113557285047) q[3];
cx q[2],q[3];
ry(1.516970111699995) q[2];
ry(-2.6614773892654275) q[3];
cx q[2],q[3];
ry(-3.1060632263510026) q[4];
ry(0.9820813885021263) q[5];
cx q[4],q[5];
ry(1.4808220079770225) q[4];
ry(-1.7538250558899877) q[5];
cx q[4],q[5];
ry(0.01846076487462422) q[6];
ry(-1.4009792276537079) q[7];
cx q[6],q[7];
ry(0.31861665529442185) q[6];
ry(-2.352295309575993) q[7];
cx q[6],q[7];
ry(-2.1833240318977447) q[0];
ry(-2.7756375808418277) q[2];
cx q[0],q[2];
ry(0.20770418672943158) q[0];
ry(-0.5308948563842116) q[2];
cx q[0],q[2];
ry(-2.853685844059639) q[2];
ry(-0.9442505052745318) q[4];
cx q[2],q[4];
ry(-0.21952844651591427) q[2];
ry(-1.1132460215430096) q[4];
cx q[2],q[4];
ry(0.18636303704011237) q[4];
ry(-0.597238670146413) q[6];
cx q[4],q[6];
ry(0.4697495257310994) q[4];
ry(-0.6053317822110024) q[6];
cx q[4],q[6];
ry(2.2127624084624875) q[1];
ry(2.9703447691562888) q[3];
cx q[1],q[3];
ry(1.147005535740459) q[1];
ry(-0.4527605563061803) q[3];
cx q[1],q[3];
ry(2.942430679540931) q[3];
ry(-1.0678053673596315) q[5];
cx q[3],q[5];
ry(-2.0017693550194684) q[3];
ry(0.8077811586412871) q[5];
cx q[3],q[5];
ry(-1.3442144364062458) q[5];
ry(1.0503525622951955) q[7];
cx q[5],q[7];
ry(1.3806034953453414) q[5];
ry(-2.601921257254564) q[7];
cx q[5],q[7];
ry(1.0558033077277416) q[0];
ry(2.5544564023641043) q[3];
cx q[0],q[3];
ry(1.283711806557089) q[0];
ry(-0.8689435231107998) q[3];
cx q[0],q[3];
ry(0.8362004395256788) q[1];
ry(0.6557275614829542) q[2];
cx q[1],q[2];
ry(-0.7793162768212412) q[1];
ry(-1.944544312076028) q[2];
cx q[1],q[2];
ry(2.3639122685518217) q[2];
ry(-2.130490032001282) q[5];
cx q[2],q[5];
ry(-1.1196251941504578) q[2];
ry(-2.135312387514082) q[5];
cx q[2],q[5];
ry(-2.6446675532391093) q[3];
ry(-2.275205965267725) q[4];
cx q[3],q[4];
ry(0.9051956170812065) q[3];
ry(1.0674079393344513) q[4];
cx q[3],q[4];
ry(-2.168463239881642) q[4];
ry(-0.35550770765711553) q[7];
cx q[4],q[7];
ry(-1.0597395978339552) q[4];
ry(0.2491887714133682) q[7];
cx q[4],q[7];
ry(2.323426616413) q[5];
ry(-2.3605013766224516) q[6];
cx q[5],q[6];
ry(1.5796642542114014) q[5];
ry(-1.793116091219554) q[6];
cx q[5],q[6];
ry(-2.017040031030252) q[0];
ry(-2.063488898930264) q[1];
cx q[0],q[1];
ry(2.7439051207920366) q[0];
ry(1.8264286073392269) q[1];
cx q[0],q[1];
ry(-0.5202958228115593) q[2];
ry(1.516674628054006) q[3];
cx q[2],q[3];
ry(-2.2045688561403134) q[2];
ry(2.0045264330278227) q[3];
cx q[2],q[3];
ry(-2.9283465469131453) q[4];
ry(2.19431093495235) q[5];
cx q[4],q[5];
ry(0.1005207183642264) q[4];
ry(-2.773078756162451) q[5];
cx q[4],q[5];
ry(0.8882488977614775) q[6];
ry(0.3170610155997959) q[7];
cx q[6],q[7];
ry(2.1572077141006223) q[6];
ry(-1.3459478398127562) q[7];
cx q[6],q[7];
ry(-0.5418191820582086) q[0];
ry(0.3920984818447732) q[2];
cx q[0],q[2];
ry(2.066430872525218) q[0];
ry(-1.176998649443025) q[2];
cx q[0],q[2];
ry(-2.8699387074580787) q[2];
ry(2.228625196863784) q[4];
cx q[2],q[4];
ry(1.0414582085903037) q[2];
ry(2.155624870282275) q[4];
cx q[2],q[4];
ry(-0.13856252640052727) q[4];
ry(-1.7243549823041864) q[6];
cx q[4],q[6];
ry(3.1123721017802297) q[4];
ry(-0.41022217164413755) q[6];
cx q[4],q[6];
ry(-0.3047281891173717) q[1];
ry(2.1630948849636527) q[3];
cx q[1],q[3];
ry(-0.4219332694470852) q[1];
ry(2.8249250020738232) q[3];
cx q[1],q[3];
ry(2.9028811782251878) q[3];
ry(-0.28894574247454485) q[5];
cx q[3],q[5];
ry(-2.1698033391185163) q[3];
ry(-2.923252141489667) q[5];
cx q[3],q[5];
ry(2.85720888756538) q[5];
ry(-0.9241044711879381) q[7];
cx q[5],q[7];
ry(2.6281132036047214) q[5];
ry(0.49992225042678484) q[7];
cx q[5],q[7];
ry(1.3159299090268615) q[0];
ry(-2.6331765203466855) q[3];
cx q[0],q[3];
ry(-2.3746322433602947) q[0];
ry(-0.42077173309464744) q[3];
cx q[0],q[3];
ry(2.464515364545415) q[1];
ry(-0.8019313549947604) q[2];
cx q[1],q[2];
ry(-1.4726768346650576) q[1];
ry(-0.9592439359406473) q[2];
cx q[1],q[2];
ry(0.6268606921213269) q[2];
ry(3.0126619427889194) q[5];
cx q[2],q[5];
ry(2.637391809051209) q[2];
ry(-1.6048736773741685) q[5];
cx q[2],q[5];
ry(2.248140435257949) q[3];
ry(-1.9013343190667498) q[4];
cx q[3],q[4];
ry(-2.0900839641916753) q[3];
ry(-2.1421044879601254) q[4];
cx q[3],q[4];
ry(-0.16553861679147558) q[4];
ry(-0.8689451446731962) q[7];
cx q[4],q[7];
ry(0.4828917644861672) q[4];
ry(-2.518974984581094) q[7];
cx q[4],q[7];
ry(0.7892407040087568) q[5];
ry(-2.8610788839254826) q[6];
cx q[5],q[6];
ry(1.8363811640579675) q[5];
ry(0.6709204838075777) q[6];
cx q[5],q[6];
ry(0.3219555740300351) q[0];
ry(-2.784918828693301) q[1];
cx q[0],q[1];
ry(-0.9966266669808954) q[0];
ry(-1.3607694840504478) q[1];
cx q[0],q[1];
ry(-0.357862532684062) q[2];
ry(0.24163342862508588) q[3];
cx q[2],q[3];
ry(2.47047372439752) q[2];
ry(0.13179479259971755) q[3];
cx q[2],q[3];
ry(0.32098424580298623) q[4];
ry(1.6872678404261672) q[5];
cx q[4],q[5];
ry(-2.7160206213980245) q[4];
ry(2.2654785617362023) q[5];
cx q[4],q[5];
ry(-2.786232668255629) q[6];
ry(1.7742937076156318) q[7];
cx q[6],q[7];
ry(-0.054301484916300505) q[6];
ry(-2.9250265814721477) q[7];
cx q[6],q[7];
ry(-1.8197173227286232) q[0];
ry(2.7114478247772342) q[2];
cx q[0],q[2];
ry(-2.8095356584394744) q[0];
ry(-1.645287725384637) q[2];
cx q[0],q[2];
ry(-2.328329377507754) q[2];
ry(0.917302311399701) q[4];
cx q[2],q[4];
ry(-0.36964156993536523) q[2];
ry(1.7444045694848453) q[4];
cx q[2],q[4];
ry(-1.3794119867266132) q[4];
ry(0.8362666018909951) q[6];
cx q[4],q[6];
ry(1.305511249861322) q[4];
ry(-0.04658094611665397) q[6];
cx q[4],q[6];
ry(2.4926165251373313) q[1];
ry(-0.7867398965880685) q[3];
cx q[1],q[3];
ry(0.8537104877244595) q[1];
ry(1.3857210283306438) q[3];
cx q[1],q[3];
ry(-1.2132130625211417) q[3];
ry(0.11919262749745839) q[5];
cx q[3],q[5];
ry(-0.28505409235775225) q[3];
ry(1.465346454944373) q[5];
cx q[3],q[5];
ry(-1.3037833725813117) q[5];
ry(-0.11872062975856666) q[7];
cx q[5],q[7];
ry(1.7471278869142541) q[5];
ry(2.0341019940245912) q[7];
cx q[5],q[7];
ry(1.8046888005778525) q[0];
ry(0.7682362610740983) q[3];
cx q[0],q[3];
ry(3.11088854801927) q[0];
ry(-2.0442243187752385) q[3];
cx q[0],q[3];
ry(0.29772525563504987) q[1];
ry(-2.6661445252545004) q[2];
cx q[1],q[2];
ry(1.7653614461254667) q[1];
ry(0.04344654356094406) q[2];
cx q[1],q[2];
ry(2.8900410953536904) q[2];
ry(0.9041481478738235) q[5];
cx q[2],q[5];
ry(2.3945858070060444) q[2];
ry(2.9285137888215407) q[5];
cx q[2],q[5];
ry(-1.88128259867604) q[3];
ry(-1.6096658675518727) q[4];
cx q[3],q[4];
ry(3.129869452967197) q[3];
ry(-1.9341304876457621) q[4];
cx q[3],q[4];
ry(2.070510785722502) q[4];
ry(1.8386038678022407) q[7];
cx q[4],q[7];
ry(0.4750734872400999) q[4];
ry(0.266590759248186) q[7];
cx q[4],q[7];
ry(1.2775466279403984) q[5];
ry(-2.3191289727222824) q[6];
cx q[5],q[6];
ry(-1.3781038474914507) q[5];
ry(1.935184762524853) q[6];
cx q[5],q[6];
ry(-1.9564496745490683) q[0];
ry(2.5150828880442426) q[1];
cx q[0],q[1];
ry(-1.5879981925431415) q[0];
ry(-1.0109851130575631) q[1];
cx q[0],q[1];
ry(-1.7040270026562396) q[2];
ry(2.5017754964593255) q[3];
cx q[2],q[3];
ry(2.4946101887835335) q[2];
ry(2.086254383516012) q[3];
cx q[2],q[3];
ry(-2.7384777194248184) q[4];
ry(0.4750067353980505) q[5];
cx q[4],q[5];
ry(2.60254674457514) q[4];
ry(-2.179701619270489) q[5];
cx q[4],q[5];
ry(-2.861231434272363) q[6];
ry(0.9846813753844206) q[7];
cx q[6],q[7];
ry(1.2393037441739105) q[6];
ry(-2.1413094177834324) q[7];
cx q[6],q[7];
ry(0.392389530306721) q[0];
ry(2.156909443974782) q[2];
cx q[0],q[2];
ry(-1.2351334774729592) q[0];
ry(-0.9120022872833831) q[2];
cx q[0],q[2];
ry(-0.33958159383739694) q[2];
ry(2.021760329409732) q[4];
cx q[2],q[4];
ry(0.37319813509313343) q[2];
ry(0.04024999753380065) q[4];
cx q[2],q[4];
ry(0.03310235274040085) q[4];
ry(0.5476029204489526) q[6];
cx q[4],q[6];
ry(-3.10813473164259) q[4];
ry(0.8860774465879926) q[6];
cx q[4],q[6];
ry(1.9810476878806629) q[1];
ry(1.49259119135374) q[3];
cx q[1],q[3];
ry(0.7372699884120201) q[1];
ry(2.263943765685064) q[3];
cx q[1],q[3];
ry(1.7270938617449347) q[3];
ry(2.3772820806400508) q[5];
cx q[3],q[5];
ry(-0.30797447179938064) q[3];
ry(-0.11361151592379316) q[5];
cx q[3],q[5];
ry(1.9005580023873732) q[5];
ry(3.004732422298404) q[7];
cx q[5],q[7];
ry(-0.5313143433754078) q[5];
ry(-1.8242446759994433) q[7];
cx q[5],q[7];
ry(-3.053367312700329) q[0];
ry(-1.6347274627805897) q[3];
cx q[0],q[3];
ry(0.5566753044750063) q[0];
ry(1.504259548083802) q[3];
cx q[0],q[3];
ry(1.9686262848994012) q[1];
ry(-2.3247011072991732) q[2];
cx q[1],q[2];
ry(-1.8653527183753358) q[1];
ry(-1.3443339338541873) q[2];
cx q[1],q[2];
ry(-0.6281815303825828) q[2];
ry(2.5844449158675133) q[5];
cx q[2],q[5];
ry(1.242903635796317) q[2];
ry(-1.4552009876594234) q[5];
cx q[2],q[5];
ry(-1.1708568193880504) q[3];
ry(-1.8491054855359743) q[4];
cx q[3],q[4];
ry(0.6508604283787891) q[3];
ry(0.8449831591939324) q[4];
cx q[3],q[4];
ry(-1.9125590115213695) q[4];
ry(-0.07925767968371122) q[7];
cx q[4],q[7];
ry(0.6909079130823805) q[4];
ry(-0.39065319339843635) q[7];
cx q[4],q[7];
ry(-1.6664995186388711) q[5];
ry(-2.4746722925787132) q[6];
cx q[5],q[6];
ry(1.244131523899892) q[5];
ry(2.970532772139733) q[6];
cx q[5],q[6];
ry(1.1913015388802177) q[0];
ry(0.857778757669947) q[1];
cx q[0],q[1];
ry(-1.7825233341731188) q[0];
ry(-2.6739076778360493) q[1];
cx q[0],q[1];
ry(-1.2447630968038252) q[2];
ry(1.004813504580481) q[3];
cx q[2],q[3];
ry(0.01907275725884173) q[2];
ry(-2.062352381824901) q[3];
cx q[2],q[3];
ry(-1.8799934132931533) q[4];
ry(-2.094027258176099) q[5];
cx q[4],q[5];
ry(2.473442658859793) q[4];
ry(2.2276734556325826) q[5];
cx q[4],q[5];
ry(1.206176833574021) q[6];
ry(-0.6583734165617754) q[7];
cx q[6],q[7];
ry(-1.9954007548429538) q[6];
ry(-0.8025328247865036) q[7];
cx q[6],q[7];
ry(0.9194311762396438) q[0];
ry(-2.413174804528271) q[2];
cx q[0],q[2];
ry(1.5790521555550399) q[0];
ry(2.587597126309587) q[2];
cx q[0],q[2];
ry(-0.6909349563573484) q[2];
ry(-0.5915519735287563) q[4];
cx q[2],q[4];
ry(-2.947319864667824) q[2];
ry(-2.179066704948876) q[4];
cx q[2],q[4];
ry(-0.9830149923127437) q[4];
ry(0.19659066062255418) q[6];
cx q[4],q[6];
ry(-0.02427406692449274) q[4];
ry(2.033062956527592) q[6];
cx q[4],q[6];
ry(0.27282910140204186) q[1];
ry(0.44289223727020577) q[3];
cx q[1],q[3];
ry(-0.2777273785878638) q[1];
ry(-1.7407816291587022) q[3];
cx q[1],q[3];
ry(-2.9747327176988265) q[3];
ry(-1.2563985932612272) q[5];
cx q[3],q[5];
ry(1.8629268964443548) q[3];
ry(2.2184222328128325) q[5];
cx q[3],q[5];
ry(-2.5522446614521046) q[5];
ry(-1.7146222512520948) q[7];
cx q[5],q[7];
ry(-2.42642551684433) q[5];
ry(1.3706177295898136) q[7];
cx q[5],q[7];
ry(-0.9259691510923123) q[0];
ry(0.14379052594708552) q[3];
cx q[0],q[3];
ry(-2.899519166294862) q[0];
ry(-1.340641649216038) q[3];
cx q[0],q[3];
ry(-2.9519877930221403) q[1];
ry(-1.906605329889348) q[2];
cx q[1],q[2];
ry(-2.9846222038569725) q[1];
ry(-2.475699594493035) q[2];
cx q[1],q[2];
ry(-1.4816635172910377) q[2];
ry(-1.1416769521606733) q[5];
cx q[2],q[5];
ry(-1.2007215404481506) q[2];
ry(2.0791437473796233) q[5];
cx q[2],q[5];
ry(-0.8690121191807305) q[3];
ry(0.1123895020905344) q[4];
cx q[3],q[4];
ry(-0.25908089194583095) q[3];
ry(-0.4214034756008793) q[4];
cx q[3],q[4];
ry(1.6371098013672665) q[4];
ry(2.029991138773325) q[7];
cx q[4],q[7];
ry(-2.125478622559664) q[4];
ry(-1.4029582642126526) q[7];
cx q[4],q[7];
ry(-1.3580922943230531) q[5];
ry(2.9827024410716083) q[6];
cx q[5],q[6];
ry(-0.2805466260215403) q[5];
ry(2.4958938743536883) q[6];
cx q[5],q[6];
ry(-0.24360650654165836) q[0];
ry(-0.22511133947423917) q[1];
cx q[0],q[1];
ry(2.142535281517903) q[0];
ry(0.6468698324865869) q[1];
cx q[0],q[1];
ry(1.0432685958962615) q[2];
ry(1.508204403427905) q[3];
cx q[2],q[3];
ry(-1.5291769131403756) q[2];
ry(-0.2640531768816494) q[3];
cx q[2],q[3];
ry(1.2096773305038742) q[4];
ry(-0.19805356814960984) q[5];
cx q[4],q[5];
ry(-2.9159356475357043) q[4];
ry(1.4694096787492192) q[5];
cx q[4],q[5];
ry(0.014167436975527359) q[6];
ry(0.5827626918409843) q[7];
cx q[6],q[7];
ry(-1.1496708587378635) q[6];
ry(1.6260749598742856) q[7];
cx q[6],q[7];
ry(0.6551716292168335) q[0];
ry(-1.128781463588161) q[2];
cx q[0],q[2];
ry(-1.2777293632573243) q[0];
ry(-2.1965013009312413) q[2];
cx q[0],q[2];
ry(-2.5696611220132652) q[2];
ry(-2.62763058818393) q[4];
cx q[2],q[4];
ry(-1.1079205336290983) q[2];
ry(-2.2304787279388902) q[4];
cx q[2],q[4];
ry(-0.7978655872123224) q[4];
ry(-2.3151882138800963) q[6];
cx q[4],q[6];
ry(2.568630524106427) q[4];
ry(0.831786034799886) q[6];
cx q[4],q[6];
ry(-2.9383029418092415) q[1];
ry(-0.6849188087433475) q[3];
cx q[1],q[3];
ry(-0.604400963007809) q[1];
ry(1.491103171489821) q[3];
cx q[1],q[3];
ry(-3.1390042985495574) q[3];
ry(-1.3459784351604018) q[5];
cx q[3],q[5];
ry(-1.7718697108305932) q[3];
ry(0.4007183399231977) q[5];
cx q[3],q[5];
ry(2.984466931512933) q[5];
ry(-0.9452441999486778) q[7];
cx q[5],q[7];
ry(0.801630250540387) q[5];
ry(-2.675233480961875) q[7];
cx q[5],q[7];
ry(-0.4685215359837406) q[0];
ry(-1.3101384664688234) q[3];
cx q[0],q[3];
ry(2.756441135213599) q[0];
ry(1.2031875813491384) q[3];
cx q[0],q[3];
ry(-1.304161591575179) q[1];
ry(-1.7280609817663144) q[2];
cx q[1],q[2];
ry(2.4817144314928243) q[1];
ry(1.9896847799374253) q[2];
cx q[1],q[2];
ry(1.112737006944027) q[2];
ry(-2.881094304507567) q[5];
cx q[2],q[5];
ry(0.5295443488762777) q[2];
ry(-3.083796129464021) q[5];
cx q[2],q[5];
ry(-2.2648375596158767) q[3];
ry(-2.456518837625886) q[4];
cx q[3],q[4];
ry(0.4011132521542553) q[3];
ry(2.836590902760075) q[4];
cx q[3],q[4];
ry(0.3355031445869319) q[4];
ry(1.6763717540832823) q[7];
cx q[4],q[7];
ry(1.7333382580943453) q[4];
ry(-1.828881180755415) q[7];
cx q[4],q[7];
ry(1.6099928159411006) q[5];
ry(0.12664307002434594) q[6];
cx q[5],q[6];
ry(-1.312885522713581) q[5];
ry(1.154195743897874) q[6];
cx q[5],q[6];
ry(-0.920213003479895) q[0];
ry(0.4071656519520426) q[1];
ry(-0.3861997289567568) q[2];
ry(2.7784439907274923) q[3];
ry(0.8069121657830278) q[4];
ry(-2.443113222355151) q[5];
ry(0.15814945576125816) q[6];
ry(-0.05160534031152598) q[7];