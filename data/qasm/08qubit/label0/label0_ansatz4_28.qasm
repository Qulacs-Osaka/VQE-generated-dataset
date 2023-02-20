OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.4078091589279995) q[0];
rz(-2.5929807447126) q[0];
ry(0.18616441225485134) q[1];
rz(0.6607245096160982) q[1];
ry(1.5089726441065971) q[2];
rz(-3.1007440366457284) q[2];
ry(0.4262996995566449) q[3];
rz(2.223087713986083) q[3];
ry(0.8755416040506028) q[4];
rz(-1.7466999781790116) q[4];
ry(-0.5394417987295839) q[5];
rz(-1.3896973077910204) q[5];
ry(-0.4654531072623697) q[6];
rz(2.3412255148287993) q[6];
ry(-0.7524727424538877) q[7];
rz(2.970760066152953) q[7];
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
ry(-0.9280261410184395) q[0];
rz(1.813727568310128) q[0];
ry(-2.28163848615568) q[1];
rz(0.8472817072594453) q[1];
ry(2.4028005749069097) q[2];
rz(-1.6349768708994048) q[2];
ry(-1.3828871045653806) q[3];
rz(-0.6781138786827885) q[3];
ry(2.5218014655552654) q[4];
rz(1.4030678937391776) q[4];
ry(-1.519333048427792) q[5];
rz(-0.656090208396713) q[5];
ry(0.7987314912554488) q[6];
rz(3.1198780840009577) q[6];
ry(1.2352432577873058) q[7];
rz(2.9584095995665582) q[7];
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
ry(-0.2355519787184304) q[0];
rz(1.6722545555594515) q[0];
ry(0.734485117319797) q[1];
rz(2.1974685061843617) q[1];
ry(0.4264550305404244) q[2];
rz(-1.7694758647292463) q[2];
ry(-2.028850279065992) q[3];
rz(0.6736053927126873) q[3];
ry(0.17938606951402036) q[4];
rz(2.2375703449116546) q[4];
ry(0.5608880993355588) q[5];
rz(2.988549955716566) q[5];
ry(-1.6903276110577736) q[6];
rz(2.4886569303374175) q[6];
ry(-0.29540874112618837) q[7];
rz(1.031033823811498) q[7];
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
ry(-0.8979792579964938) q[0];
rz(-1.1051223291310697) q[0];
ry(-2.276678762920172) q[1];
rz(-1.628753650667751) q[1];
ry(0.9835717853658476) q[2];
rz(0.11433520214594672) q[2];
ry(0.6423476011499227) q[3];
rz(-1.503160690554011) q[3];
ry(-0.12589117318509935) q[4];
rz(1.216784965009383) q[4];
ry(-2.4170524812926466) q[5];
rz(3.0594795970889295) q[5];
ry(-0.4637789630655762) q[6];
rz(0.5330718127534421) q[6];
ry(0.3877150127450074) q[7];
rz(2.2631055285340778) q[7];
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
ry(2.21215272582854) q[0];
rz(-1.4920725164330781) q[0];
ry(-0.7576137022227468) q[1];
rz(2.0642780257825097) q[1];
ry(-3.1175639363007344) q[2];
rz(-2.896004262742068) q[2];
ry(-1.1706210720894719) q[3];
rz(-2.4778140206225743) q[3];
ry(-2.0630530289065128) q[4];
rz(2.7100413140731265) q[4];
ry(-1.065639981527453) q[5];
rz(-2.4770243293491965) q[5];
ry(-1.9109737187120093) q[6];
rz(-1.6170151802215909) q[6];
ry(-2.9339063769826272) q[7];
rz(-2.888670536264784) q[7];
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
ry(-0.9075797226956795) q[0];
rz(-3.100157433640596) q[0];
ry(1.3349696080373459) q[1];
rz(-1.6120536527887221) q[1];
ry(-1.2003120206116678) q[2];
rz(-0.46295984661356704) q[2];
ry(2.3503477980762586) q[3];
rz(2.8749359187666244) q[3];
ry(-0.29540143896712845) q[4];
rz(0.3018099916326893) q[4];
ry(0.19062685330107063) q[5];
rz(1.946512647440949) q[5];
ry(-0.7045347576459856) q[6];
rz(-2.6178756025987004) q[6];
ry(3.0120284860223627) q[7];
rz(-0.14662950353170154) q[7];
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
ry(0.22145055244968323) q[0];
rz(-1.273863454703452) q[0];
ry(0.7611268553247477) q[1];
rz(-2.3433433476507814) q[1];
ry(-1.5680831067815744) q[2];
rz(-1.9978738612237121) q[2];
ry(-0.3698034118568088) q[3];
rz(-2.1563352862891567) q[3];
ry(-0.28274995033821954) q[4];
rz(0.4389446901674264) q[4];
ry(0.45008753871861007) q[5];
rz(-0.03572118923757815) q[5];
ry(-2.369552334267255) q[6];
rz(0.522811613620692) q[6];
ry(-2.403892551587073) q[7];
rz(1.0993244190976208) q[7];
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
ry(-2.823604486454222) q[0];
rz(2.649011242509448) q[0];
ry(1.0188320576593648) q[1];
rz(0.8117117040391673) q[1];
ry(1.1255454564082648) q[2];
rz(0.12222804471382553) q[2];
ry(2.969039588026232) q[3];
rz(1.5143000191056926) q[3];
ry(0.6947451968387117) q[4];
rz(2.975635131118076) q[4];
ry(-1.4892922617154198) q[5];
rz(-1.972170263856264) q[5];
ry(-1.3136898663864056) q[6];
rz(-1.7646830962933961) q[6];
ry(-1.1210618888195913) q[7];
rz(-2.7898180904630756) q[7];
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
ry(0.30266233151694166) q[0];
rz(2.2716823094870966) q[0];
ry(2.8908862637323263) q[1];
rz(-3.1089132442949734) q[1];
ry(1.2430907998931886) q[2];
rz(-0.9621581744825948) q[2];
ry(0.9683298751499858) q[3];
rz(-2.380055851640189) q[3];
ry(-0.20996766827122215) q[4];
rz(3.0546428104065835) q[4];
ry(-1.785999470506117) q[5];
rz(2.917750912763189) q[5];
ry(-1.0233802791252726) q[6];
rz(-0.27063358194143045) q[6];
ry(0.3778592686968638) q[7];
rz(1.0738624566401669) q[7];
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
ry(1.5194861129176556) q[0];
rz(2.5134268342994663) q[0];
ry(-2.017287834169327) q[1];
rz(1.6259491505914632) q[1];
ry(1.9908197324210075) q[2];
rz(2.611308397680824) q[2];
ry(2.878155109010035) q[3];
rz(-1.3640966241010868) q[3];
ry(2.6636570537205895) q[4];
rz(2.0410731153092194) q[4];
ry(0.909240506909647) q[5];
rz(0.3297451220779329) q[5];
ry(2.258034076967427) q[6];
rz(1.0442075523706178) q[6];
ry(2.3190446532777367) q[7];
rz(1.163870160205212) q[7];
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
ry(-2.562244429620933) q[0];
rz(2.765155301142107) q[0];
ry(-0.41857045752456434) q[1];
rz(-1.7658313039632434) q[1];
ry(2.0452674801953163) q[2];
rz(-1.5370562355692519) q[2];
ry(-0.8763794238754429) q[3];
rz(-2.7504441629921645) q[3];
ry(2.410249362450885) q[4];
rz(-0.9115944202744851) q[4];
ry(1.873189305728741) q[5];
rz(2.449352148210778) q[5];
ry(1.7006531272504901) q[6];
rz(1.9726588407362975) q[6];
ry(-1.6668104137244062) q[7];
rz(-1.6049546765167302) q[7];
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
ry(2.489470908894987) q[0];
rz(3.04833927654457) q[0];
ry(1.7565556404469966) q[1];
rz(-0.02619244019962515) q[1];
ry(3.0417167384068855) q[2];
rz(-2.3107782456214343) q[2];
ry(-2.855586350939343) q[3];
rz(-0.6214124414258653) q[3];
ry(-0.6283956600145295) q[4];
rz(-2.3407374661443088) q[4];
ry(-1.4657533507484315) q[5];
rz(2.757420016783001) q[5];
ry(1.0402454898001974) q[6];
rz(1.1171808582079492) q[6];
ry(0.1425410729994594) q[7];
rz(-0.9662967070925541) q[7];
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
ry(1.2513932512152603) q[0];
rz(-0.8076811979559976) q[0];
ry(-1.448182526367881) q[1];
rz(-1.140014894491436) q[1];
ry(1.9642885996232637) q[2];
rz(-1.307929797303948) q[2];
ry(-2.0860627376197947) q[3];
rz(-0.1548578787234626) q[3];
ry(1.8004126618843588) q[4];
rz(-2.724016102936168) q[4];
ry(2.744031525236287) q[5];
rz(2.6542408064425898) q[5];
ry(-2.7604016704395096) q[6];
rz(0.5188339934635087) q[6];
ry(2.9699434913862244) q[7];
rz(-1.406705570836412) q[7];
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
ry(3.084464073445558) q[0];
rz(1.3811764783193679) q[0];
ry(2.639693581229129) q[1];
rz(-2.218319450150244) q[1];
ry(-0.39262313201534466) q[2];
rz(-2.3607973468287202) q[2];
ry(0.966869996655807) q[3];
rz(1.5961120900166654) q[3];
ry(-1.259993825848106) q[4];
rz(-2.43317844921598) q[4];
ry(0.2798173318976316) q[5];
rz(2.2602269754399202) q[5];
ry(-2.617024924235392) q[6];
rz(2.741105322929155) q[6];
ry(2.9285282937778714) q[7];
rz(-3.1352823817000095) q[7];
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
ry(-2.159448825781935) q[0];
rz(1.5967644183162086) q[0];
ry(-1.112388833020358) q[1];
rz(-1.0304356988043895) q[1];
ry(-2.1404571640250687) q[2];
rz(-1.7122462162735628) q[2];
ry(-2.83350393947792) q[3];
rz(0.042768433504298604) q[3];
ry(-2.999599422754996) q[4];
rz(2.7708232487620483) q[4];
ry(1.8088094395811858) q[5];
rz(1.7439349217660458) q[5];
ry(-1.4253890029021623) q[6];
rz(-2.6230704723640312) q[6];
ry(2.463217550602724) q[7];
rz(2.0175472661692715) q[7];
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
ry(-2.8176432584075455) q[0];
rz(-2.1062537962379384) q[0];
ry(-0.6131138362114575) q[1];
rz(-3.0049891100851522) q[1];
ry(1.7553640593475155) q[2];
rz(0.49990134305147116) q[2];
ry(-1.2560691991540451) q[3];
rz(-1.654252834212885) q[3];
ry(0.8392977842615769) q[4];
rz(-1.5795675347181213) q[4];
ry(1.4387682464276388) q[5];
rz(1.48484915982778) q[5];
ry(0.3919473444879902) q[6];
rz(2.3384509720108086) q[6];
ry(-1.6370333450075294) q[7];
rz(2.4005592165394325) q[7];
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
ry(-2.735351268248524) q[0];
rz(-1.5263638005954312) q[0];
ry(-0.900856997302891) q[1];
rz(-2.9082940170982265) q[1];
ry(-0.05469212639210142) q[2];
rz(2.055417332066086) q[2];
ry(1.8358112454143907) q[3];
rz(2.48119192711351) q[3];
ry(0.25689769744406826) q[4];
rz(-0.1117723584094481) q[4];
ry(2.3831510108192444) q[5];
rz(-0.10739163861107888) q[5];
ry(2.9900157860830223) q[6];
rz(2.8538124386340313) q[6];
ry(-0.11307593600693709) q[7];
rz(2.7963704645517855) q[7];
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
ry(2.612006324632827) q[0];
rz(-2.1654741877488073) q[0];
ry(-0.25946169049492546) q[1];
rz(-2.465526640193726) q[1];
ry(-2.528949677688847) q[2];
rz(-1.2716924495247377) q[2];
ry(0.7015038233876145) q[3];
rz(1.2474013309433825) q[3];
ry(-0.7829511802524802) q[4];
rz(-0.0009390825197971437) q[4];
ry(0.15724198300266326) q[5];
rz(-2.828279824912951) q[5];
ry(1.1689623246513356) q[6];
rz(1.3738865401121483) q[6];
ry(1.2172251817175601) q[7];
rz(-2.4688281087503086) q[7];
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
ry(-3.130362470528542) q[0];
rz(2.7920092642151033) q[0];
ry(-1.2230268765705328) q[1];
rz(2.0487366861360865) q[1];
ry(0.4876535377568221) q[2];
rz(-2.1695785138163854) q[2];
ry(-0.30000363161149624) q[3];
rz(-2.6880869331116823) q[3];
ry(2.0118974068414563) q[4];
rz(-2.9218223107676127) q[4];
ry(1.644594673987621) q[5];
rz(-1.4541308155705857) q[5];
ry(0.4469943800393764) q[6];
rz(1.1254633106491918) q[6];
ry(2.298509703623185) q[7];
rz(-1.6451366732210717) q[7];
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
ry(2.3464434073953497) q[0];
rz(2.3555536215638266) q[0];
ry(3.0651652466793275) q[1];
rz(0.3508322845186074) q[1];
ry(0.15730907028226504) q[2];
rz(-2.0133910187387336) q[2];
ry(-2.508560991411618) q[3];
rz(1.5254182244446126) q[3];
ry(-0.7711279120865774) q[4];
rz(-2.371229084589203) q[4];
ry(-3.026029166314012) q[5];
rz(-3.04902158000211) q[5];
ry(0.18647839189505497) q[6];
rz(2.681402673058071) q[6];
ry(-2.9636829293572493) q[7];
rz(1.75307827880469) q[7];
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
ry(2.372103587232509) q[0];
rz(1.3371737025120656) q[0];
ry(1.6297787860538033) q[1];
rz(3.1399953838951813) q[1];
ry(1.4928287318921005) q[2];
rz(-0.48836932183129994) q[2];
ry(1.5350032585441375) q[3];
rz(0.3055150006098373) q[3];
ry(2.058027656320629) q[4];
rz(1.0477773084860882) q[4];
ry(1.6753014016295174) q[5];
rz(-1.0052143202864008) q[5];
ry(2.4026009043569414) q[6];
rz(-1.3551850622772734) q[6];
ry(1.3847912675782212) q[7];
rz(3.1159401207452393) q[7];
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
ry(-1.287610669682416) q[0];
rz(1.297111999135668) q[0];
ry(2.7665336943774386) q[1];
rz(1.661054419529865) q[1];
ry(-2.742248353084625) q[2];
rz(0.5204869549838094) q[2];
ry(-1.5005014961944394) q[3];
rz(-2.3865753750654757) q[3];
ry(1.8699155185462584) q[4];
rz(0.32609897349006184) q[4];
ry(-1.7820527681929323) q[5];
rz(-2.356544194440463) q[5];
ry(0.8893032320858183) q[6];
rz(-2.6304521023935523) q[6];
ry(-0.7808760881478919) q[7];
rz(1.4830200928047592) q[7];
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
ry(0.4342389937300691) q[0];
rz(2.021757197132554) q[0];
ry(-1.9016240577220134) q[1];
rz(1.1690713327566609) q[1];
ry(-0.7729391164303155) q[2];
rz(-2.7839566370654274) q[2];
ry(1.227133511869174) q[3];
rz(1.6654536151122012) q[3];
ry(-0.9162479882942112) q[4];
rz(-0.7158723474202207) q[4];
ry(-3.0483161613796965) q[5];
rz(-0.5315012126035015) q[5];
ry(0.6326974640607528) q[6];
rz(-0.6905993256124053) q[6];
ry(0.6652878426230789) q[7];
rz(2.7254991482406443) q[7];
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
ry(0.2843662942451213) q[0];
rz(0.8030468225874321) q[0];
ry(1.8787092836544819) q[1];
rz(2.506375679695148) q[1];
ry(-0.24598418577631637) q[2];
rz(-2.3777957146615596) q[2];
ry(-0.33850225912121523) q[3];
rz(-0.1475430950211912) q[3];
ry(0.9194092078858733) q[4];
rz(3.0567577624254487) q[4];
ry(-1.4516400450859062) q[5];
rz(1.2525016106843985) q[5];
ry(0.7717261824983446) q[6];
rz(-0.7559046477184276) q[6];
ry(-1.470637009923787) q[7];
rz(-1.643128463912999) q[7];
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
ry(-1.41676100485525) q[0];
rz(1.888011343907138) q[0];
ry(2.9502415949770753) q[1];
rz(0.988319878612736) q[1];
ry(-1.3509833539124028) q[2];
rz(-1.4241722250397473) q[2];
ry(2.3738741602659386) q[3];
rz(-1.0053847348958576) q[3];
ry(0.46524188670684796) q[4];
rz(0.12633754334073288) q[4];
ry(0.04643932562316578) q[5];
rz(1.8697920387820544) q[5];
ry(2.609755831360348) q[6];
rz(-1.1389428364811454) q[6];
ry(1.351400566002396) q[7];
rz(1.2813609113681386) q[7];
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
ry(-0.4384764319373888) q[0];
rz(-0.3547881652807554) q[0];
ry(0.17147684264345625) q[1];
rz(2.959002350743042) q[1];
ry(2.275650064630172) q[2];
rz(-1.9883372827834365) q[2];
ry(2.3529584313711873) q[3];
rz(-1.8418892618188711) q[3];
ry(-1.6580109326895558) q[4];
rz(-2.317818540584505) q[4];
ry(-2.920393925906968) q[5];
rz(2.1036946890265193) q[5];
ry(-1.3261699097453903) q[6];
rz(0.41828875180017605) q[6];
ry(-0.3356953457690763) q[7];
rz(-1.3539389322806432) q[7];
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
ry(-2.4901204707107647) q[0];
rz(1.7825447612445928) q[0];
ry(2.517961984752387) q[1];
rz(-0.8187929486865199) q[1];
ry(0.30660370705318074) q[2];
rz(-0.9995030345247188) q[2];
ry(-0.8784836565123335) q[3];
rz(-1.0457381232424363) q[3];
ry(-1.3629368499262124) q[4];
rz(0.7113127078614746) q[4];
ry(-0.25714176125919813) q[5];
rz(-2.678870724543787) q[5];
ry(-0.5908121315207768) q[6];
rz(-1.0846039261320874) q[6];
ry(2.6507639228679984) q[7];
rz(1.811646329027167) q[7];
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
ry(2.7077863455562188) q[0];
rz(3.08904394110631) q[0];
ry(2.126628485434129) q[1];
rz(1.1086772761476418) q[1];
ry(-0.81622010831434) q[2];
rz(1.4313427099996219) q[2];
ry(-1.52315555292472) q[3];
rz(-2.12293640728281) q[3];
ry(0.9029865605651352) q[4];
rz(1.1933951728386694) q[4];
ry(2.311998424788809) q[5];
rz(0.7432667620564359) q[5];
ry(1.8547898773267355) q[6];
rz(-0.13260388861988928) q[6];
ry(-0.7568672559262521) q[7];
rz(-2.8821808348315883) q[7];
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
ry(-1.9018309443225643) q[0];
rz(-0.451950844964097) q[0];
ry(2.293303987200008) q[1];
rz(2.2493088939076062) q[1];
ry(-2.1379585096424694) q[2];
rz(-1.6710523450972916) q[2];
ry(2.7380879836199594) q[3];
rz(2.0145074131020717) q[3];
ry(-2.303536731384292) q[4];
rz(0.9990328183775663) q[4];
ry(1.5194433009495265) q[5];
rz(1.2284300355684987) q[5];
ry(0.6906921616330655) q[6];
rz(0.3973964846553319) q[6];
ry(-2.3827871090548856) q[7];
rz(2.7466490595462862) q[7];
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
ry(-1.3071851267794667) q[0];
rz(-1.3867102883398825) q[0];
ry(-1.6876293252074044) q[1];
rz(1.3666598675251214) q[1];
ry(1.4697176728795665) q[2];
rz(2.235755767835677) q[2];
ry(-3.0613750774907973) q[3];
rz(0.04157254997084347) q[3];
ry(1.6653259321424305) q[4];
rz(-0.998716743514258) q[4];
ry(-0.3398099717228291) q[5];
rz(-0.8735472247364432) q[5];
ry(-0.8677880007095762) q[6];
rz(-0.5713348285813993) q[6];
ry(-1.5289404637290573) q[7];
rz(2.9375838867189055) q[7];
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
ry(0.9148921012452931) q[0];
rz(-0.12082973523140074) q[0];
ry(-1.9228079014052852) q[1];
rz(0.8488330444990178) q[1];
ry(1.592961336675871) q[2];
rz(2.031386490819256) q[2];
ry(0.4093371178881335) q[3];
rz(-2.125854279828654) q[3];
ry(2.3623629659752066) q[4];
rz(-1.6515966748966402) q[4];
ry(-0.6357404620912062) q[5];
rz(-1.0013368945067482) q[5];
ry(2.7799035704088944) q[6];
rz(2.474908110431052) q[6];
ry(1.7842453590354115) q[7];
rz(-0.7098074813715954) q[7];
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
ry(-2.3146709064322017) q[0];
rz(-0.2866773457397803) q[0];
ry(-0.18685502746467206) q[1];
rz(1.6360949876807611) q[1];
ry(-1.5731890365996015) q[2];
rz(2.914692914571511) q[2];
ry(-0.7199157440830739) q[3];
rz(-2.585361495964415) q[3];
ry(1.5020958389554737) q[4];
rz(-2.878626297180786) q[4];
ry(-0.6455099258677395) q[5];
rz(2.6932782960210364) q[5];
ry(-1.2605403158774129) q[6];
rz(2.0125685077009123) q[6];
ry(2.0544776213353675) q[7];
rz(0.9759094623385138) q[7];