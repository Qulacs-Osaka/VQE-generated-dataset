OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.3704448178672461) q[0];
ry(-2.90285026096381) q[1];
cx q[0],q[1];
ry(1.9636010023194004) q[0];
ry(-3.0775516830844163) q[1];
cx q[0],q[1];
ry(-2.0837295013560926) q[2];
ry(-0.6352313169798893) q[3];
cx q[2],q[3];
ry(-0.0675895595567465) q[2];
ry(3.1151891829745226) q[3];
cx q[2],q[3];
ry(-0.42772284734758576) q[1];
ry(1.3069330980058664) q[2];
cx q[1],q[2];
ry(-0.8144247350059852) q[1];
ry(0.7660395241995717) q[2];
cx q[1],q[2];
ry(-1.7766739561581284) q[0];
ry(0.015880297049987016) q[1];
cx q[0],q[1];
ry(1.5417586968389114) q[0];
ry(1.0767310591521015) q[1];
cx q[0],q[1];
ry(1.4429087547918775) q[2];
ry(-1.6837912355789204) q[3];
cx q[2],q[3];
ry(0.2227503220042113) q[2];
ry(-0.08791882587474208) q[3];
cx q[2],q[3];
ry(0.5370336380576273) q[1];
ry(-0.3656253963583151) q[2];
cx q[1],q[2];
ry(-0.1995996797786216) q[1];
ry(2.468838269283316) q[2];
cx q[1],q[2];
ry(0.8477574856374974) q[0];
ry(2.777267549435341) q[1];
cx q[0],q[1];
ry(-1.5537760578856465) q[0];
ry(2.878987955652318) q[1];
cx q[0],q[1];
ry(-2.4099530761115844) q[2];
ry(1.8409368236258512) q[3];
cx q[2],q[3];
ry(-2.400281254615561) q[2];
ry(-2.8502874997714063) q[3];
cx q[2],q[3];
ry(-2.577476206963796) q[1];
ry(-0.2831642922586529) q[2];
cx q[1],q[2];
ry(2.9738912463848797) q[1];
ry(1.237427210584013) q[2];
cx q[1],q[2];
ry(-0.9510019566092539) q[0];
ry(0.8790025572560356) q[1];
cx q[0],q[1];
ry(2.6227713337883545) q[0];
ry(-2.9410895458201116) q[1];
cx q[0],q[1];
ry(-2.7730806084802664) q[2];
ry(-2.6364259261900265) q[3];
cx q[2],q[3];
ry(1.1023366540529105) q[2];
ry(-1.7587988085632185) q[3];
cx q[2],q[3];
ry(-1.252989962846685) q[1];
ry(0.054679342107120554) q[2];
cx q[1],q[2];
ry(0.5790066992816048) q[1];
ry(0.7251978998059059) q[2];
cx q[1],q[2];
ry(1.2301662149337858) q[0];
ry(-3.0912459761304207) q[1];
cx q[0],q[1];
ry(-2.8379307049813223) q[0];
ry(-1.5231162666657205) q[1];
cx q[0],q[1];
ry(-1.5555370707078966) q[2];
ry(-1.7958477390534897) q[3];
cx q[2],q[3];
ry(1.9925214495378227) q[2];
ry(0.04361415255891892) q[3];
cx q[2],q[3];
ry(-0.2456802842910184) q[1];
ry(0.8398200536743222) q[2];
cx q[1],q[2];
ry(-0.2952024449503359) q[1];
ry(2.5534191238523696) q[2];
cx q[1],q[2];
ry(1.6074213648718843) q[0];
ry(1.2111990080110147) q[1];
cx q[0],q[1];
ry(2.66839631834042) q[0];
ry(2.352302064081515) q[1];
cx q[0],q[1];
ry(0.20312543736917732) q[2];
ry(-0.1921258676398936) q[3];
cx q[2],q[3];
ry(1.6993392797995774) q[2];
ry(-2.134946824486918) q[3];
cx q[2],q[3];
ry(-1.5985450138065491) q[1];
ry(0.8367366963377192) q[2];
cx q[1],q[2];
ry(-0.11928714356400819) q[1];
ry(2.3321894632337243) q[2];
cx q[1],q[2];
ry(-0.02341421631915565) q[0];
ry(-1.9409596721920392) q[1];
cx q[0],q[1];
ry(-2.56788081449892) q[0];
ry(-1.0592062853899602) q[1];
cx q[0],q[1];
ry(-1.5201661300979765) q[2];
ry(2.730128498743655) q[3];
cx q[2],q[3];
ry(-0.15189945729212534) q[2];
ry(-0.9579313010221587) q[3];
cx q[2],q[3];
ry(0.7699073236275638) q[1];
ry(2.8544550419260477) q[2];
cx q[1],q[2];
ry(-1.11287419013447) q[1];
ry(2.5754058620353537) q[2];
cx q[1],q[2];
ry(2.1614687886805855) q[0];
ry(0.34511086058719265) q[1];
cx q[0],q[1];
ry(2.7589490217234522) q[0];
ry(-1.9993015019152107) q[1];
cx q[0],q[1];
ry(0.07208439798501498) q[2];
ry(2.4325301380539988) q[3];
cx q[2],q[3];
ry(1.1909294060911422) q[2];
ry(0.2938302040706926) q[3];
cx q[2],q[3];
ry(-1.3362867061237518) q[1];
ry(-2.1345494360580624) q[2];
cx q[1],q[2];
ry(1.3290155449656904) q[1];
ry(-1.832434972843702) q[2];
cx q[1],q[2];
ry(-0.43510407947141516) q[0];
ry(-0.9994561136065734) q[1];
cx q[0],q[1];
ry(-0.12103636942757666) q[0];
ry(-2.712210789684935) q[1];
cx q[0],q[1];
ry(-2.8152515508784246) q[2];
ry(0.49670600230636697) q[3];
cx q[2],q[3];
ry(0.8482373291161374) q[2];
ry(-0.723444719445369) q[3];
cx q[2],q[3];
ry(-3.114021154510873) q[1];
ry(-1.078521883206796) q[2];
cx q[1],q[2];
ry(0.6163790490908195) q[1];
ry(-1.463665645371913) q[2];
cx q[1],q[2];
ry(-1.7276752291212203) q[0];
ry(0.3550359296174106) q[1];
cx q[0],q[1];
ry(-2.216907461216978) q[0];
ry(1.1107588995186513) q[1];
cx q[0],q[1];
ry(2.3603793931421158) q[2];
ry(1.1360235517487385) q[3];
cx q[2],q[3];
ry(-1.604615619745141) q[2];
ry(1.9755272234200714) q[3];
cx q[2],q[3];
ry(-3.0878625005540328) q[1];
ry(-0.22273557686508205) q[2];
cx q[1],q[2];
ry(-1.211730024454158) q[1];
ry(2.7858092266210157) q[2];
cx q[1],q[2];
ry(1.6538161625807475) q[0];
ry(1.9090043725620598) q[1];
cx q[0],q[1];
ry(-0.8956598684326685) q[0];
ry(0.7103002365212338) q[1];
cx q[0],q[1];
ry(1.1957781607137896) q[2];
ry(0.9792725733027172) q[3];
cx q[2],q[3];
ry(0.33260352020715456) q[2];
ry(-2.8490455563514616) q[3];
cx q[2],q[3];
ry(0.38127885511586135) q[1];
ry(2.6009366932873808) q[2];
cx q[1],q[2];
ry(-2.6666177749349114) q[1];
ry(1.247680607260819) q[2];
cx q[1],q[2];
ry(2.4098200331465485) q[0];
ry(2.6155022100089442) q[1];
cx q[0],q[1];
ry(0.33645100540398926) q[0];
ry(1.642360518895984) q[1];
cx q[0],q[1];
ry(2.198341619815251) q[2];
ry(-2.315039179008161) q[3];
cx q[2],q[3];
ry(-1.8222042109094165) q[2];
ry(1.977634499038507) q[3];
cx q[2],q[3];
ry(0.12497394974351347) q[1];
ry(2.9659286218652055) q[2];
cx q[1],q[2];
ry(1.0349053662472985) q[1];
ry(2.8220678843750986) q[2];
cx q[1],q[2];
ry(2.0953234846074205) q[0];
ry(2.018961913404207) q[1];
cx q[0],q[1];
ry(0.4857148063481002) q[0];
ry(-2.7911077061579737) q[1];
cx q[0],q[1];
ry(-1.0903313698793515) q[2];
ry(-0.9138749151537955) q[3];
cx q[2],q[3];
ry(2.0125213547142353) q[2];
ry(-1.0932111229135222) q[3];
cx q[2],q[3];
ry(0.4745044299615564) q[1];
ry(-1.987520820781369) q[2];
cx q[1],q[2];
ry(-2.6807629063328644) q[1];
ry(-2.9050500311942518) q[2];
cx q[1],q[2];
ry(2.689385172554762) q[0];
ry(-0.9835970535519587) q[1];
cx q[0],q[1];
ry(-1.035213842189223) q[0];
ry(0.8167697475356817) q[1];
cx q[0],q[1];
ry(-0.7059716022611277) q[2];
ry(-0.9682274152625094) q[3];
cx q[2],q[3];
ry(2.882016973439002) q[2];
ry(-1.3573548283690409) q[3];
cx q[2],q[3];
ry(2.357769226324194) q[1];
ry(-3.0698516906370292) q[2];
cx q[1],q[2];
ry(-2.4182646121232523) q[1];
ry(1.388543935384434) q[2];
cx q[1],q[2];
ry(-1.2236225780389312) q[0];
ry(2.720346017381476) q[1];
cx q[0],q[1];
ry(0.6072337566583105) q[0];
ry(-0.8889890348180207) q[1];
cx q[0],q[1];
ry(3.0736067560722447) q[2];
ry(-0.3866762041834804) q[3];
cx q[2],q[3];
ry(-2.728235286304885) q[2];
ry(-1.5484222453277) q[3];
cx q[2],q[3];
ry(-0.907309389957029) q[1];
ry(2.1416055740905584) q[2];
cx q[1],q[2];
ry(-0.6546396800100152) q[1];
ry(-2.474427639752527) q[2];
cx q[1],q[2];
ry(1.3938413364942752) q[0];
ry(0.4073054039651822) q[1];
ry(0.6078287814335468) q[2];
ry(-2.7170156943236456) q[3];