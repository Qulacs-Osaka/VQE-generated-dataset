OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.41797251855365475) q[0];
ry(0.11281856224875109) q[1];
cx q[0],q[1];
ry(-2.517903903456375) q[0];
ry(0.03281046908028884) q[1];
cx q[0],q[1];
ry(0.5043887769791773) q[2];
ry(1.0358043206197325) q[3];
cx q[2],q[3];
ry(-1.4785456090864981) q[2];
ry(2.422765121718306) q[3];
cx q[2],q[3];
ry(-1.081728299685346) q[4];
ry(-0.32308980619286665) q[5];
cx q[4],q[5];
ry(0.13063579827737468) q[4];
ry(0.2591651856239361) q[5];
cx q[4],q[5];
ry(-2.4647916225983595) q[6];
ry(1.5603039024141012) q[7];
cx q[6],q[7];
ry(-2.596221723770048) q[6];
ry(-0.009394628934036742) q[7];
cx q[6],q[7];
ry(2.6595816898296905) q[0];
ry(2.7093654404298024) q[2];
cx q[0],q[2];
ry(0.3006931956997967) q[0];
ry(1.570754707155884) q[2];
cx q[0],q[2];
ry(1.1114109326543171) q[2];
ry(-3.1378395846656733) q[4];
cx q[2],q[4];
ry(-0.5435275308557073) q[2];
ry(1.5967700380270236) q[4];
cx q[2],q[4];
ry(-1.8270024853048716) q[4];
ry(2.5452886052119847) q[6];
cx q[4],q[6];
ry(2.248954470009974) q[4];
ry(1.2953007511877672) q[6];
cx q[4],q[6];
ry(1.1064541180583147) q[1];
ry(-2.8099272852505135) q[3];
cx q[1],q[3];
ry(-2.0541553922343017) q[1];
ry(2.7499470634870353) q[3];
cx q[1],q[3];
ry(0.5130606737054784) q[3];
ry(-0.5884013817176807) q[5];
cx q[3],q[5];
ry(2.3142834622260757) q[3];
ry(2.431291336696439) q[5];
cx q[3],q[5];
ry(-0.8892587915729847) q[5];
ry(-3.047591320251529) q[7];
cx q[5],q[7];
ry(-0.8403110142141287) q[5];
ry(2.025879474876197) q[7];
cx q[5],q[7];
ry(-2.1924039714259203) q[0];
ry(-3.0059724764188243) q[1];
cx q[0],q[1];
ry(1.9404054843154732) q[0];
ry(1.6530355774747507) q[1];
cx q[0],q[1];
ry(1.8789694052124046) q[2];
ry(2.2667551321578463) q[3];
cx q[2],q[3];
ry(0.6571180955878182) q[2];
ry(2.3187923314765455) q[3];
cx q[2],q[3];
ry(-0.9787099762251457) q[4];
ry(-1.3030073995333407) q[5];
cx q[4],q[5];
ry(-1.1191122544308585) q[4];
ry(0.9811553292236105) q[5];
cx q[4],q[5];
ry(3.0037926435744273) q[6];
ry(1.1202913779768189) q[7];
cx q[6],q[7];
ry(1.0528753224339198) q[6];
ry(-1.4708418216047656) q[7];
cx q[6],q[7];
ry(-1.4481438015044814) q[0];
ry(2.138029319188905) q[2];
cx q[0],q[2];
ry(-1.4670608880654195) q[0];
ry(1.2365622246156702) q[2];
cx q[0],q[2];
ry(-0.9352444774128185) q[2];
ry(-2.450026844363589) q[4];
cx q[2],q[4];
ry(-0.0895060801556416) q[2];
ry(-1.4728401513590645) q[4];
cx q[2],q[4];
ry(-2.753382066612018) q[4];
ry(-0.06329114842236372) q[6];
cx q[4],q[6];
ry(1.942592671779563) q[4];
ry(3.060055574330631) q[6];
cx q[4],q[6];
ry(1.946748800731375) q[1];
ry(0.2864321207398777) q[3];
cx q[1],q[3];
ry(0.436653902394289) q[1];
ry(-1.6902270310567653) q[3];
cx q[1],q[3];
ry(-2.9343985620086803) q[3];
ry(2.4928096799843336) q[5];
cx q[3],q[5];
ry(2.9191434180050213) q[3];
ry(-2.979517142765252) q[5];
cx q[3],q[5];
ry(1.9688394300022143) q[5];
ry(0.703960806184536) q[7];
cx q[5],q[7];
ry(1.9917882402043219) q[5];
ry(3.0688261878236145) q[7];
cx q[5],q[7];
ry(0.8101991178835483) q[0];
ry(1.4121939821607405) q[1];
cx q[0],q[1];
ry(-1.4367126851930885) q[0];
ry(-0.12429377584338434) q[1];
cx q[0],q[1];
ry(-1.069162956327399) q[2];
ry(2.2574916987731575) q[3];
cx q[2],q[3];
ry(2.1880550658076885) q[2];
ry(0.5770799102148754) q[3];
cx q[2],q[3];
ry(-2.593441287288655) q[4];
ry(-0.9142310996617785) q[5];
cx q[4],q[5];
ry(2.6484790334450734) q[4];
ry(1.8402552108050934) q[5];
cx q[4],q[5];
ry(0.22460732292309699) q[6];
ry(-1.8600996932440284) q[7];
cx q[6],q[7];
ry(-1.4307230835166043) q[6];
ry(0.09658737018178432) q[7];
cx q[6],q[7];
ry(-0.42122916929110055) q[0];
ry(-2.0101373484322473) q[2];
cx q[0],q[2];
ry(0.47037584811804556) q[0];
ry(-2.631948197461624) q[2];
cx q[0],q[2];
ry(1.3043094587546493) q[2];
ry(-2.910695241072238) q[4];
cx q[2],q[4];
ry(2.139396927827371) q[2];
ry(1.4878469363146076) q[4];
cx q[2],q[4];
ry(-2.7070012050588637) q[4];
ry(1.675907993861217) q[6];
cx q[4],q[6];
ry(0.23504499615505153) q[4];
ry(-0.17495713739135563) q[6];
cx q[4],q[6];
ry(1.27173765237238) q[1];
ry(1.673548474525366) q[3];
cx q[1],q[3];
ry(2.2396201315922166) q[1];
ry(1.8955409024181877) q[3];
cx q[1],q[3];
ry(-0.5704846372909298) q[3];
ry(0.5650300292419788) q[5];
cx q[3],q[5];
ry(0.17382900489869876) q[3];
ry(1.1640012599740892) q[5];
cx q[3],q[5];
ry(-0.425868824721535) q[5];
ry(2.2446374718458726) q[7];
cx q[5],q[7];
ry(-2.820044709925201) q[5];
ry(1.5152714810603216) q[7];
cx q[5],q[7];
ry(1.7825648310481843) q[0];
ry(-1.1827079998742174) q[1];
cx q[0],q[1];
ry(0.6271365415233897) q[0];
ry(1.5715355897979713) q[1];
cx q[0],q[1];
ry(-0.788993549373989) q[2];
ry(-2.5872799223970495) q[3];
cx q[2],q[3];
ry(-1.6444238421138275) q[2];
ry(2.5531325920714583) q[3];
cx q[2],q[3];
ry(1.259396881187488) q[4];
ry(2.8441860093113713) q[5];
cx q[4],q[5];
ry(-0.25451591937569606) q[4];
ry(-0.4112090946996515) q[5];
cx q[4],q[5];
ry(-2.2996224461603383) q[6];
ry(-2.9022559246239195) q[7];
cx q[6],q[7];
ry(1.1609205959749822) q[6];
ry(-0.5173358164805049) q[7];
cx q[6],q[7];
ry(-2.601401571920836) q[0];
ry(0.42183224736301905) q[2];
cx q[0],q[2];
ry(-0.32291400334196935) q[0];
ry(-1.3348437892942497) q[2];
cx q[0],q[2];
ry(2.98665405860322) q[2];
ry(-1.0802372992606752) q[4];
cx q[2],q[4];
ry(1.2116104522631677) q[2];
ry(1.2041758099313196) q[4];
cx q[2],q[4];
ry(-2.601162596253251) q[4];
ry(2.234743468553874) q[6];
cx q[4],q[6];
ry(0.5084393323467928) q[4];
ry(-1.325683571693988) q[6];
cx q[4],q[6];
ry(0.5567526406047341) q[1];
ry(2.483459885448988) q[3];
cx q[1],q[3];
ry(2.535692147506202) q[1];
ry(-2.5789521934960935) q[3];
cx q[1],q[3];
ry(2.1639938811366353) q[3];
ry(-1.2928032330467056) q[5];
cx q[3],q[5];
ry(-0.9341091646664264) q[3];
ry(1.5319453385406216) q[5];
cx q[3],q[5];
ry(0.008660366579961044) q[5];
ry(-0.11053152359007713) q[7];
cx q[5],q[7];
ry(1.0534590171881693) q[5];
ry(-0.4339688328364479) q[7];
cx q[5],q[7];
ry(1.2089308880215162) q[0];
ry(-0.40202245054527364) q[1];
cx q[0],q[1];
ry(-1.1778756970946187) q[0];
ry(-0.9201667898036366) q[1];
cx q[0],q[1];
ry(-0.8610401252466735) q[2];
ry(-1.3447357457743774) q[3];
cx q[2],q[3];
ry(2.6711019883413365) q[2];
ry(-0.8312433287890625) q[3];
cx q[2],q[3];
ry(-1.8657293268511932) q[4];
ry(1.1120992628216515) q[5];
cx q[4],q[5];
ry(2.9840497060925872) q[4];
ry(-0.17510604467812196) q[5];
cx q[4],q[5];
ry(3.101684594438684) q[6];
ry(-1.7209745612205662) q[7];
cx q[6],q[7];
ry(0.4048450448063222) q[6];
ry(-0.4679102763364112) q[7];
cx q[6],q[7];
ry(0.7720448031903944) q[0];
ry(0.603318573239016) q[2];
cx q[0],q[2];
ry(-3.1248636154914897) q[0];
ry(1.3781365250510618) q[2];
cx q[0],q[2];
ry(-1.388232202046114) q[2];
ry(-0.7499046344741717) q[4];
cx q[2],q[4];
ry(0.570509649493034) q[2];
ry(2.074478000445497) q[4];
cx q[2],q[4];
ry(0.3499554641603497) q[4];
ry(-0.11605147239254167) q[6];
cx q[4],q[6];
ry(-0.5832176956580062) q[4];
ry(-1.4958507347013867) q[6];
cx q[4],q[6];
ry(2.1418408038140906) q[1];
ry(1.6702116699762843) q[3];
cx q[1],q[3];
ry(-2.0381139829112547) q[1];
ry(2.1764425865644954) q[3];
cx q[1],q[3];
ry(2.5976994391224744) q[3];
ry(-2.8806786760069483) q[5];
cx q[3],q[5];
ry(0.22846974382445634) q[3];
ry(1.068909521316912) q[5];
cx q[3],q[5];
ry(-2.2397325445954683) q[5];
ry(-2.3526692764903694) q[7];
cx q[5],q[7];
ry(-2.8993824625219533) q[5];
ry(0.13033927105806065) q[7];
cx q[5],q[7];
ry(-1.1316678768447133) q[0];
ry(-1.472843775881898) q[1];
cx q[0],q[1];
ry(-3.1340495024084274) q[0];
ry(-0.7697729551066417) q[1];
cx q[0],q[1];
ry(0.2689420163789542) q[2];
ry(-0.5686268224906436) q[3];
cx q[2],q[3];
ry(1.9230308354758474) q[2];
ry(-1.663349109526741) q[3];
cx q[2],q[3];
ry(-1.9453984642058673) q[4];
ry(0.2196508939891082) q[5];
cx q[4],q[5];
ry(-0.744696912977421) q[4];
ry(0.9905793699404076) q[5];
cx q[4],q[5];
ry(-2.9282449069770404) q[6];
ry(0.5684545598803213) q[7];
cx q[6],q[7];
ry(0.44255941006413685) q[6];
ry(3.0967397675830304) q[7];
cx q[6],q[7];
ry(-0.0317687377784309) q[0];
ry(1.1654865925813331) q[2];
cx q[0],q[2];
ry(-0.10735868060147169) q[0];
ry(-2.5482652323001145) q[2];
cx q[0],q[2];
ry(1.077589076226129) q[2];
ry(-0.8231577908318058) q[4];
cx q[2],q[4];
ry(-2.7911664419629227) q[2];
ry(-2.4441864084340112) q[4];
cx q[2],q[4];
ry(-2.277131145559572) q[4];
ry(-2.4372735092583784) q[6];
cx q[4],q[6];
ry(-0.7188071686488112) q[4];
ry(0.039071121387550534) q[6];
cx q[4],q[6];
ry(0.2240516889172423) q[1];
ry(0.5337530672613642) q[3];
cx q[1],q[3];
ry(1.1489154386478528) q[1];
ry(2.4196052093353084) q[3];
cx q[1],q[3];
ry(-3.0991805944055115) q[3];
ry(-1.749763863776497) q[5];
cx q[3],q[5];
ry(0.865002075986065) q[3];
ry(2.9831444404828815) q[5];
cx q[3],q[5];
ry(0.23156886802264123) q[5];
ry(1.8213054862988942) q[7];
cx q[5],q[7];
ry(-1.0580888573763643) q[5];
ry(-2.264880729164335) q[7];
cx q[5],q[7];
ry(1.568719302177851) q[0];
ry(3.058411552988166) q[1];
cx q[0],q[1];
ry(-1.982342033231708) q[0];
ry(1.108675339911679) q[1];
cx q[0],q[1];
ry(2.8441617963833976) q[2];
ry(-2.9056269520017155) q[3];
cx q[2],q[3];
ry(-1.10888856311768) q[2];
ry(0.6857568787331614) q[3];
cx q[2],q[3];
ry(-3.131828244293406) q[4];
ry(0.2691925496198966) q[5];
cx q[4],q[5];
ry(1.9852495532895762) q[4];
ry(1.8463259197167574) q[5];
cx q[4],q[5];
ry(-2.9448569445703976) q[6];
ry(0.14422041027114307) q[7];
cx q[6],q[7];
ry(-0.7786832480737272) q[6];
ry(0.5722870029076635) q[7];
cx q[6],q[7];
ry(-0.6414667091732529) q[0];
ry(-0.34723097828237753) q[2];
cx q[0],q[2];
ry(-0.3846845334689684) q[0];
ry(-1.5012014728966365) q[2];
cx q[0],q[2];
ry(-0.40037767378007433) q[2];
ry(1.6888381285296923) q[4];
cx q[2],q[4];
ry(-1.4603983470714814) q[2];
ry(-1.1303889499725477) q[4];
cx q[2],q[4];
ry(1.303042193302949) q[4];
ry(-1.4564796791093357) q[6];
cx q[4],q[6];
ry(2.596721444812737) q[4];
ry(2.641759142840833) q[6];
cx q[4],q[6];
ry(-1.1248948084121961) q[1];
ry(-1.907534839925324) q[3];
cx q[1],q[3];
ry(-2.494239284945636) q[1];
ry(0.993159329147023) q[3];
cx q[1],q[3];
ry(2.163109821663359) q[3];
ry(-0.058386203390823316) q[5];
cx q[3],q[5];
ry(2.7024219324626464) q[3];
ry(2.61605565293373) q[5];
cx q[3],q[5];
ry(-2.626736483536188) q[5];
ry(-0.7861256870415513) q[7];
cx q[5],q[7];
ry(-3.106499621098295) q[5];
ry(0.10197708961957244) q[7];
cx q[5],q[7];
ry(0.08268062135925369) q[0];
ry(-2.7443269394178085) q[1];
cx q[0],q[1];
ry(3.0396960666645807) q[0];
ry(3.070080539831486) q[1];
cx q[0],q[1];
ry(0.44315288192398183) q[2];
ry(2.7852568301626763) q[3];
cx q[2],q[3];
ry(2.9175667244618184) q[2];
ry(-1.3269797442728997) q[3];
cx q[2],q[3];
ry(-0.30210422792627156) q[4];
ry(3.040204429161444) q[5];
cx q[4],q[5];
ry(-0.21504314599636487) q[4];
ry(0.10226711598233111) q[5];
cx q[4],q[5];
ry(-1.7448272763777983) q[6];
ry(-2.478436325882683) q[7];
cx q[6],q[7];
ry(1.7459781109778738) q[6];
ry(3.071374797849914) q[7];
cx q[6],q[7];
ry(-0.5063395560383663) q[0];
ry(2.8878473024465405) q[2];
cx q[0],q[2];
ry(2.7050015643735157) q[0];
ry(1.6802789494794586) q[2];
cx q[0],q[2];
ry(-0.49329903071131703) q[2];
ry(-0.0023237504999231717) q[4];
cx q[2],q[4];
ry(3.03720129540066) q[2];
ry(1.7506948464050085) q[4];
cx q[2],q[4];
ry(-0.18953934272647888) q[4];
ry(2.758182227257129) q[6];
cx q[4],q[6];
ry(-2.351563108603092) q[4];
ry(-0.28907594158003663) q[6];
cx q[4],q[6];
ry(1.4882801116888054) q[1];
ry(1.1321036053602227) q[3];
cx q[1],q[3];
ry(2.2274252034629543) q[1];
ry(0.2096404367167933) q[3];
cx q[1],q[3];
ry(-1.6624469472998635) q[3];
ry(2.679643365815682) q[5];
cx q[3],q[5];
ry(2.9594808854978223) q[3];
ry(-2.6835395981965515) q[5];
cx q[3],q[5];
ry(2.692878335679673) q[5];
ry(2.7469241239139053) q[7];
cx q[5],q[7];
ry(2.027657520810566) q[5];
ry(-0.42460450209492606) q[7];
cx q[5],q[7];
ry(3.0259227040307) q[0];
ry(2.3879294305127923) q[1];
cx q[0],q[1];
ry(2.8965754893324736) q[0];
ry(-1.1520343084386342) q[1];
cx q[0],q[1];
ry(2.26668979997683) q[2];
ry(0.8568031092338213) q[3];
cx q[2],q[3];
ry(-1.824044731052602) q[2];
ry(1.047516222285818) q[3];
cx q[2],q[3];
ry(-1.8620390986468982) q[4];
ry(0.1128048027188292) q[5];
cx q[4],q[5];
ry(3.1263148963328486) q[4];
ry(-0.6557054185978056) q[5];
cx q[4],q[5];
ry(1.5633532649727764) q[6];
ry(2.3039493970276914) q[7];
cx q[6],q[7];
ry(-2.17795246321466) q[6];
ry(1.9773687503418413) q[7];
cx q[6],q[7];
ry(0.6885974586226987) q[0];
ry(0.7155188529917326) q[2];
cx q[0],q[2];
ry(0.548278196680811) q[0];
ry(2.411290555590391) q[2];
cx q[0],q[2];
ry(-3.0256796787101994) q[2];
ry(-2.1110726019645156) q[4];
cx q[2],q[4];
ry(2.5753192677803973) q[2];
ry(-1.3076015415340416) q[4];
cx q[2],q[4];
ry(1.7700977189710068) q[4];
ry(-2.1585438324487267) q[6];
cx q[4],q[6];
ry(-2.587935313259289) q[4];
ry(-0.527273251416119) q[6];
cx q[4],q[6];
ry(2.074176162885254) q[1];
ry(-0.04563425273311983) q[3];
cx q[1],q[3];
ry(0.0406963884071363) q[1];
ry(-2.476511182687623) q[3];
cx q[1],q[3];
ry(-1.5590414859784647) q[3];
ry(-0.8049952112935513) q[5];
cx q[3],q[5];
ry(-0.7245447816641916) q[3];
ry(1.0041043803743248) q[5];
cx q[3],q[5];
ry(2.4419021081251544) q[5];
ry(-1.7954464307736553) q[7];
cx q[5],q[7];
ry(-2.2156325314312335) q[5];
ry(-2.4830391524302615) q[7];
cx q[5],q[7];
ry(-0.47005911782371124) q[0];
ry(1.5980744529302138) q[1];
cx q[0],q[1];
ry(-1.0646821539007734) q[0];
ry(-0.7658495270341203) q[1];
cx q[0],q[1];
ry(2.62187452084482) q[2];
ry(0.27314871995092116) q[3];
cx q[2],q[3];
ry(2.203076464094041) q[2];
ry(-1.4584656186212428) q[3];
cx q[2],q[3];
ry(0.39966362265856786) q[4];
ry(2.891026780318016) q[5];
cx q[4],q[5];
ry(1.9806133059501692) q[4];
ry(1.3711134634930167) q[5];
cx q[4],q[5];
ry(1.6939363501518852) q[6];
ry(1.2391213243490993) q[7];
cx q[6],q[7];
ry(-0.4402094283679172) q[6];
ry(2.7182925329616143) q[7];
cx q[6],q[7];
ry(0.7710187322334345) q[0];
ry(-1.449247561209825) q[2];
cx q[0],q[2];
ry(0.9735266215421836) q[0];
ry(-1.8018139506504711) q[2];
cx q[0],q[2];
ry(0.690372275985343) q[2];
ry(-0.7854929596557233) q[4];
cx q[2],q[4];
ry(-1.0792390315711602) q[2];
ry(-2.5499112171710547) q[4];
cx q[2],q[4];
ry(-1.5312393578025736) q[4];
ry(-0.9565644362490043) q[6];
cx q[4],q[6];
ry(-1.4396752354219335) q[4];
ry(-2.9642512315526464) q[6];
cx q[4],q[6];
ry(2.720345352310629) q[1];
ry(1.0147886808793665) q[3];
cx q[1],q[3];
ry(2.4008299256328804) q[1];
ry(-1.6354860751618434) q[3];
cx q[1],q[3];
ry(-2.8685003136604306) q[3];
ry(-1.666194056207302) q[5];
cx q[3],q[5];
ry(1.346860694049262) q[3];
ry(0.5708229272831558) q[5];
cx q[3],q[5];
ry(1.720488644895037) q[5];
ry(1.8544437285636135) q[7];
cx q[5],q[7];
ry(1.0833433829706522) q[5];
ry(-1.4013727318617306) q[7];
cx q[5],q[7];
ry(0.4258662846516883) q[0];
ry(-0.2740404999269682) q[1];
cx q[0],q[1];
ry(2.9105630279461137) q[0];
ry(2.2604549550643647) q[1];
cx q[0],q[1];
ry(2.5544936226526116) q[2];
ry(-2.7302563930593164) q[3];
cx q[2],q[3];
ry(-1.5899908985148064) q[2];
ry(-0.6696246145794452) q[3];
cx q[2],q[3];
ry(-1.4560953966318628) q[4];
ry(2.811531028530737) q[5];
cx q[4],q[5];
ry(0.14127870127461065) q[4];
ry(0.11736713787558184) q[5];
cx q[4],q[5];
ry(-0.9752554807163621) q[6];
ry(-1.9984532911466077) q[7];
cx q[6],q[7];
ry(2.2298720405364056) q[6];
ry(3.108964346184621) q[7];
cx q[6],q[7];
ry(1.720405228419314) q[0];
ry(-0.26751307347634623) q[2];
cx q[0],q[2];
ry(-1.4660402554140062) q[0];
ry(2.8076752651159724) q[2];
cx q[0],q[2];
ry(-0.5301050830187188) q[2];
ry(0.9416175173491911) q[4];
cx q[2],q[4];
ry(1.425454014843334) q[2];
ry(-2.075494064828644) q[4];
cx q[2],q[4];
ry(1.1204563943084906) q[4];
ry(2.3352158726205916) q[6];
cx q[4],q[6];
ry(-0.23415674167463615) q[4];
ry(0.7499576232282065) q[6];
cx q[4],q[6];
ry(3.057578183924074) q[1];
ry(1.9197264508133502) q[3];
cx q[1],q[3];
ry(-2.2785045324942423) q[1];
ry(2.9305035417345446) q[3];
cx q[1],q[3];
ry(-1.815534227285783) q[3];
ry(1.2150933469389118) q[5];
cx q[3],q[5];
ry(-0.19572706719921398) q[3];
ry(-2.3130714076562384) q[5];
cx q[3],q[5];
ry(-0.25991729881587017) q[5];
ry(-1.3428398499606882) q[7];
cx q[5],q[7];
ry(3.096905472181495) q[5];
ry(2.2622396828366877) q[7];
cx q[5],q[7];
ry(-0.5682608712637264) q[0];
ry(0.08870576837028121) q[1];
cx q[0],q[1];
ry(-2.7254346853908316) q[0];
ry(1.6242929315843215) q[1];
cx q[0],q[1];
ry(1.25276009516364) q[2];
ry(0.3820677227990079) q[3];
cx q[2],q[3];
ry(-1.815885657894781) q[2];
ry(2.934846177379196) q[3];
cx q[2],q[3];
ry(2.2443703504018977) q[4];
ry(-0.3575738910683697) q[5];
cx q[4],q[5];
ry(-3.129331470700135) q[4];
ry(-2.389164998273246) q[5];
cx q[4],q[5];
ry(-2.057754515100454) q[6];
ry(-0.6910820111485627) q[7];
cx q[6],q[7];
ry(1.5370255468948502) q[6];
ry(-1.9030894301443562) q[7];
cx q[6],q[7];
ry(-0.9674140041025588) q[0];
ry(-1.7442797011921911) q[2];
cx q[0],q[2];
ry(-3.060479806758259) q[0];
ry(-2.725978723769935) q[2];
cx q[0],q[2];
ry(-1.1455844550430492) q[2];
ry(-2.261784272692494) q[4];
cx q[2],q[4];
ry(1.475297279134512) q[2];
ry(-2.604839844266901) q[4];
cx q[2],q[4];
ry(2.9592450585498296) q[4];
ry(-1.3705177565215958) q[6];
cx q[4],q[6];
ry(1.463401794395326) q[4];
ry(-2.7243829042460463) q[6];
cx q[4],q[6];
ry(2.3887906956011986) q[1];
ry(2.004941685151482) q[3];
cx q[1],q[3];
ry(0.5181048357560529) q[1];
ry(1.2239081817285675) q[3];
cx q[1],q[3];
ry(1.710671107162676) q[3];
ry(0.24010634621709137) q[5];
cx q[3],q[5];
ry(1.125140242773182) q[3];
ry(-0.08548140693156735) q[5];
cx q[3],q[5];
ry(-2.8357652331843153) q[5];
ry(1.5466170603667297) q[7];
cx q[5],q[7];
ry(-1.0259603457361677) q[5];
ry(-0.9506661654771236) q[7];
cx q[5],q[7];
ry(-0.26469243697082057) q[0];
ry(-1.529232639552466) q[1];
cx q[0],q[1];
ry(-1.2301186772475243) q[0];
ry(-1.8320292103354323) q[1];
cx q[0],q[1];
ry(1.3969610483525192) q[2];
ry(-1.0061757868733086) q[3];
cx q[2],q[3];
ry(2.036360481482858) q[2];
ry(-2.439385559150958) q[3];
cx q[2],q[3];
ry(-1.4220982301542255) q[4];
ry(0.13603052599371335) q[5];
cx q[4],q[5];
ry(2.2545085665360194) q[4];
ry(-0.59662289696994) q[5];
cx q[4],q[5];
ry(1.971179324936386) q[6];
ry(-0.2889708549458945) q[7];
cx q[6],q[7];
ry(-0.6602518920432447) q[6];
ry(1.2189135058678646) q[7];
cx q[6],q[7];
ry(1.1844195311717334) q[0];
ry(1.6152818891887177) q[2];
cx q[0],q[2];
ry(-2.098447355883855) q[0];
ry(-0.4971060983450677) q[2];
cx q[0],q[2];
ry(0.24893262795025278) q[2];
ry(1.6510299844428644) q[4];
cx q[2],q[4];
ry(0.23295760218220174) q[2];
ry(-3.0737695441759914) q[4];
cx q[2],q[4];
ry(-2.885594084539547) q[4];
ry(2.3180643656017694) q[6];
cx q[4],q[6];
ry(-0.009155898367144444) q[4];
ry(-1.0024677390645342) q[6];
cx q[4],q[6];
ry(-3.0065739032863577) q[1];
ry(-2.482916181947801) q[3];
cx q[1],q[3];
ry(1.6897396524070203) q[1];
ry(-1.3570721405307729) q[3];
cx q[1],q[3];
ry(0.15127173393953036) q[3];
ry(-1.4807577491740949) q[5];
cx q[3],q[5];
ry(1.165595041856239) q[3];
ry(2.7954032478861786) q[5];
cx q[3],q[5];
ry(1.708151892579474) q[5];
ry(-0.03691651588786015) q[7];
cx q[5],q[7];
ry(0.8592112951650871) q[5];
ry(-0.3498142041505656) q[7];
cx q[5],q[7];
ry(-0.5981569329651499) q[0];
ry(-2.1900941856940266) q[1];
cx q[0],q[1];
ry(-2.139882446736224) q[0];
ry(-1.9497313180937865) q[1];
cx q[0],q[1];
ry(1.7797330933852864) q[2];
ry(-1.1011434695646134) q[3];
cx q[2],q[3];
ry(-2.5796939340683513) q[2];
ry(2.3194496336890005) q[3];
cx q[2],q[3];
ry(2.9632348728689695) q[4];
ry(2.885389518188428) q[5];
cx q[4],q[5];
ry(-2.2794438841049045) q[4];
ry(-1.1932925273794384) q[5];
cx q[4],q[5];
ry(0.019433245867203797) q[6];
ry(0.3781375723970566) q[7];
cx q[6],q[7];
ry(-2.705716986442958) q[6];
ry(-1.642697206536886) q[7];
cx q[6],q[7];
ry(-2.649211152405408) q[0];
ry(-1.7781804717047214) q[2];
cx q[0],q[2];
ry(0.7309101891839002) q[0];
ry(1.1207003354928318) q[2];
cx q[0],q[2];
ry(1.3778724846878703) q[2];
ry(-1.5712918566446612) q[4];
cx q[2],q[4];
ry(-1.6759692291016766) q[2];
ry(1.5103894003789833) q[4];
cx q[2],q[4];
ry(2.010653636248068) q[4];
ry(3.07563968964028) q[6];
cx q[4],q[6];
ry(0.3363805181760827) q[4];
ry(-2.452572981162037) q[6];
cx q[4],q[6];
ry(2.517118875660707) q[1];
ry(1.5214970487706783) q[3];
cx q[1],q[3];
ry(-2.9612549564095234) q[1];
ry(0.9620859344928681) q[3];
cx q[1],q[3];
ry(-0.589977136737815) q[3];
ry(2.354874926137579) q[5];
cx q[3],q[5];
ry(-2.0306361585330075) q[3];
ry(-1.9154836035905127) q[5];
cx q[3],q[5];
ry(2.2366078780134364) q[5];
ry(-0.0033428820038281515) q[7];
cx q[5],q[7];
ry(2.8434395591000925) q[5];
ry(-1.033038263404336) q[7];
cx q[5],q[7];
ry(1.4701811425900382) q[0];
ry(-0.6333725749455085) q[1];
cx q[0],q[1];
ry(-1.54033814031302) q[0];
ry(-2.384227504370393) q[1];
cx q[0],q[1];
ry(-1.9705681738375085) q[2];
ry(-1.0016651089074262) q[3];
cx q[2],q[3];
ry(1.307713963362028) q[2];
ry(-0.596199985472107) q[3];
cx q[2],q[3];
ry(0.060189130935866864) q[4];
ry(-2.868527371496962) q[5];
cx q[4],q[5];
ry(2.664409917537085) q[4];
ry(-2.6779064525525254) q[5];
cx q[4],q[5];
ry(2.2411641185447424) q[6];
ry(-1.0436092478924737) q[7];
cx q[6],q[7];
ry(-2.1228843564461224) q[6];
ry(2.8255250968587258) q[7];
cx q[6],q[7];
ry(3.000492108308638) q[0];
ry(2.8282778615039397) q[2];
cx q[0],q[2];
ry(-0.4729877882558726) q[0];
ry(0.3543416323845072) q[2];
cx q[0],q[2];
ry(1.8874496934634397) q[2];
ry(-1.2185859722496701) q[4];
cx q[2],q[4];
ry(-2.9063038109719135) q[2];
ry(-2.875275096318018) q[4];
cx q[2],q[4];
ry(-0.2100102379086355) q[4];
ry(-3.017877220970684) q[6];
cx q[4],q[6];
ry(1.2420000015390873) q[4];
ry(1.9367908002406153) q[6];
cx q[4],q[6];
ry(-1.9499740604006819) q[1];
ry(1.4592357519263217) q[3];
cx q[1],q[3];
ry(-2.5554132921623283) q[1];
ry(0.7792386068506911) q[3];
cx q[1],q[3];
ry(-0.24541242331019422) q[3];
ry(2.435672137709721) q[5];
cx q[3],q[5];
ry(-1.5211191580139272) q[3];
ry(0.9619700822836573) q[5];
cx q[3],q[5];
ry(-0.1614255839435027) q[5];
ry(-1.552132978561903) q[7];
cx q[5],q[7];
ry(-0.12670918244505733) q[5];
ry(-0.5652840299471821) q[7];
cx q[5],q[7];
ry(2.2723215461023565) q[0];
ry(1.6792739600236433) q[1];
ry(-2.4523677123326886) q[2];
ry(0.4001595067972819) q[3];
ry(2.922999311529598) q[4];
ry(-0.7522316334593699) q[5];
ry(0.04068111972537694) q[6];
ry(2.340761047835339) q[7];