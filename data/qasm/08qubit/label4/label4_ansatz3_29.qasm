OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.775919094893394) q[0];
rz(-1.1577325105507137) q[0];
ry(-2.48731655108545) q[1];
rz(1.5096443761525435) q[1];
ry(1.2029423251514704) q[2];
rz(0.49400704678798046) q[2];
ry(-0.07936301628379017) q[3];
rz(2.144114372586653) q[3];
ry(-2.04115356196351) q[4];
rz(-2.9650626459091063) q[4];
ry(2.454150758903003) q[5];
rz(2.9874435793443554) q[5];
ry(2.5278153907496264) q[6];
rz(-2.0019906262548828) q[6];
ry(-0.17800504869112466) q[7];
rz(2.581849070691589) q[7];
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
ry(-1.0576560076288992) q[0];
rz(1.5670176464984555) q[0];
ry(-1.7138550553769658) q[1];
rz(-1.2118399647865616) q[1];
ry(-0.4539024747650995) q[2];
rz(-1.13899886393828) q[2];
ry(2.9010909823160214) q[3];
rz(-0.1487657549754395) q[3];
ry(-3.0535727799865113) q[4];
rz(-0.85153009227976) q[4];
ry(1.8964279899178087) q[5];
rz(-0.016586662397932276) q[5];
ry(-2.936745899525793) q[6];
rz(0.3755791651675233) q[6];
ry(1.4487705844859189) q[7];
rz(1.9725883512105575) q[7];
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
ry(1.1934866901592616) q[0];
rz(-1.0760557086341338) q[0];
ry(-1.533336605072635) q[1];
rz(2.5603934295837236) q[1];
ry(-2.230374725509111) q[2];
rz(-1.1291174081793567) q[2];
ry(1.0311930772349918) q[3];
rz(-1.3108833618626665) q[3];
ry(2.8491566523160126) q[4];
rz(2.1657248848016564) q[4];
ry(-0.9482084517468824) q[5];
rz(-1.473953873737352) q[5];
ry(2.8976213345548687) q[6];
rz(1.0315728679094756) q[6];
ry(-2.4871931163863317) q[7];
rz(-2.023058846204175) q[7];
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
ry(-2.687334538636777) q[0];
rz(-3.015791613378213) q[0];
ry(-0.9670549810918576) q[1];
rz(1.629520169361804) q[1];
ry(0.8976063658019884) q[2];
rz(1.7704883450581619) q[2];
ry(1.3971946793825163) q[3];
rz(2.8065437048278485) q[3];
ry(1.2824547195904101) q[4];
rz(-0.14470874962376282) q[4];
ry(-2.0050541093597003) q[5];
rz(0.3602392761926101) q[5];
ry(2.693286947684122) q[6];
rz(1.4130027653920028) q[6];
ry(-2.829365849257619) q[7];
rz(1.6407971606409442) q[7];
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
ry(2.5118911603553222) q[0];
rz(-2.022368936480235) q[0];
ry(-3.1372715629330394) q[1];
rz(2.6068252038265913) q[1];
ry(-1.96623720935135) q[2];
rz(-0.7749811112389579) q[2];
ry(2.0384261693737806) q[3];
rz(-2.0319294621363415) q[3];
ry(-2.7078570800829835) q[4];
rz(0.4347662335570855) q[4];
ry(-1.273628957062282) q[5];
rz(1.7071640412005185) q[5];
ry(-2.467129893023884) q[6];
rz(-2.028395368139562) q[6];
ry(-2.413141615174248) q[7];
rz(-2.2257070026635457) q[7];
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
ry(0.7470176015772522) q[0];
rz(-0.9911644093349157) q[0];
ry(1.429560959600379) q[1];
rz(-0.5862051960231511) q[1];
ry(1.0388507106779956) q[2];
rz(-0.9412964259870433) q[2];
ry(0.3632548446449053) q[3];
rz(1.1467888953206513) q[3];
ry(-1.2580698749478765) q[4];
rz(1.01711754500039) q[4];
ry(-0.5157788877325471) q[5];
rz(1.1678049710332754) q[5];
ry(-0.9542592130593331) q[6];
rz(1.8035428724623248) q[6];
ry(0.5274616377217614) q[7];
rz(0.6171387351899362) q[7];
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
ry(1.1054132659477196) q[0];
rz(1.1343653051497034) q[0];
ry(0.3353457947059243) q[1];
rz(2.0542655856331864) q[1];
ry(0.3501255949300602) q[2];
rz(-1.989363771410079) q[2];
ry(-2.1701156568405953) q[3];
rz(-1.5165240995061928) q[3];
ry(-1.8322111414643212) q[4];
rz(-1.8566002834236472) q[4];
ry(0.8042238519899154) q[5];
rz(2.2934390963929823) q[5];
ry(-3.1062294352698108) q[6];
rz(2.800283310567684) q[6];
ry(0.5311546972248741) q[7];
rz(-2.3776939659742466) q[7];
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
ry(-2.3920360680110657) q[0];
rz(2.9059623821085965) q[0];
ry(-2.0020121096999457) q[1];
rz(-2.8334548338916323) q[1];
ry(-1.0548132720281098) q[2];
rz(3.0539837151392177) q[2];
ry(-1.345737160701917) q[3];
rz(0.4573171315432856) q[3];
ry(2.174035935269871) q[4];
rz(1.8247082255171865) q[4];
ry(-0.676089730386459) q[5];
rz(0.7799404461639874) q[5];
ry(1.4133418094891201) q[6];
rz(-2.3047459885603883) q[6];
ry(-1.768454427097086) q[7];
rz(1.600610967357875) q[7];
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
ry(0.4295622435939367) q[0];
rz(3.0999789109378333) q[0];
ry(0.7322243065436296) q[1];
rz(-2.2719493764473118) q[1];
ry(2.6155914333563337) q[2];
rz(0.8734040379043717) q[2];
ry(-1.6355583845271147) q[3];
rz(1.150364560868776) q[3];
ry(-0.5216400801817027) q[4];
rz(-1.8862179712848706) q[4];
ry(0.8165159866552643) q[5];
rz(1.5859253686038297) q[5];
ry(2.0478502616439913) q[6];
rz(-1.951908682202879) q[6];
ry(0.10267255633327947) q[7];
rz(-0.6538912068281908) q[7];
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
ry(0.544222593401546) q[0];
rz(0.9610133313442483) q[0];
ry(1.6364623337378164) q[1];
rz(1.014467458216101) q[1];
ry(-2.672398233564805) q[2];
rz(1.4043404086134783) q[2];
ry(1.9874272350088942) q[3];
rz(0.36749859798363366) q[3];
ry(-2.337481397262117) q[4];
rz(-2.342824760310606) q[4];
ry(2.3508265510606825) q[5];
rz(-2.6019646743748805) q[5];
ry(-2.814901827443182) q[6];
rz(-2.3068572337154625) q[6];
ry(-0.1882574286726388) q[7];
rz(1.8557581050125) q[7];
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
ry(2.887898878944421) q[0];
rz(-1.3679952936604405) q[0];
ry(2.6642179716628394) q[1];
rz(-2.44774745241725) q[1];
ry(0.971034570320426) q[2];
rz(-2.679031013546402) q[2];
ry(-0.39191458207093266) q[3];
rz(-0.6587680302317838) q[3];
ry(0.8672608960865432) q[4];
rz(1.4795605902056206) q[4];
ry(2.4761318071772638) q[5];
rz(0.23355133215950305) q[5];
ry(-1.0829191617966398) q[6];
rz(0.1908468972870878) q[6];
ry(-0.7634408129644503) q[7];
rz(1.3303762418436105) q[7];
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
ry(1.0108570596392288) q[0];
rz(-0.5372297050329162) q[0];
ry(-2.2541399242515077) q[1];
rz(1.0559854870618115) q[1];
ry(-3.0855718010230557) q[2];
rz(1.8474249895214918) q[2];
ry(-2.6746861044785915) q[3];
rz(1.6167949329109377) q[3];
ry(2.8365770708808333) q[4];
rz(2.51843783297372) q[4];
ry(0.3077620417154633) q[5];
rz(1.3221260107302313) q[5];
ry(-0.9237660862740329) q[6];
rz(-1.6688717196773757) q[6];
ry(0.8233794963101353) q[7];
rz(-2.102661203101207) q[7];
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
ry(-2.0510317180951194) q[0];
rz(-3.0346666667530227) q[0];
ry(-1.2160322317068504) q[1];
rz(2.1596899892664148) q[1];
ry(-0.2045199101684284) q[2];
rz(0.007321936872260221) q[2];
ry(0.1646134824903287) q[3];
rz(0.8458785194976531) q[3];
ry(-2.146394567385749) q[4];
rz(1.3927889925138934) q[4];
ry(1.3782868989967378) q[5];
rz(-1.4946837197358862) q[5];
ry(-0.3683973369284076) q[6];
rz(-0.7240145478974309) q[6];
ry(-1.9132979258654972) q[7];
rz(0.4957461148718956) q[7];
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
ry(2.5697796260612518) q[0];
rz(-2.7969799546237217) q[0];
ry(-1.6049607785835445) q[1];
rz(1.9215511967447743) q[1];
ry(-1.9260931144183768) q[2];
rz(2.0145623747126287) q[2];
ry(-2.2237775379581493) q[3];
rz(-3.071600524987163) q[3];
ry(1.4890970407329214) q[4];
rz(2.4387230533715343) q[4];
ry(-2.0160556794104023) q[5];
rz(0.2937884548709868) q[5];
ry(-0.735402055475527) q[6];
rz(0.3798371715267645) q[6];
ry(2.623594983125187) q[7];
rz(-0.4950594823143897) q[7];
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
ry(2.1822745945649555) q[0];
rz(2.9096240183531394) q[0];
ry(0.9815886694294473) q[1];
rz(0.9736376839666272) q[1];
ry(-0.40585661176676713) q[2];
rz(1.8734212059192012) q[2];
ry(-2.3272995206542086) q[3];
rz(1.8279397354814204) q[3];
ry(1.5801589151139717) q[4];
rz(-1.1166920998828722) q[4];
ry(-1.454658807008544) q[5];
rz(2.201211085194843) q[5];
ry(0.20697455313629515) q[6];
rz(2.265495757753734) q[6];
ry(2.596036259667321) q[7];
rz(1.7862171612995112) q[7];
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
ry(-3.082631841395174) q[0];
rz(-2.8582440107414575) q[0];
ry(-1.2856785105442203) q[1];
rz(-2.01178458978982) q[1];
ry(-2.1316319002484923) q[2];
rz(-0.13054978287713934) q[2];
ry(2.8999318203857753) q[3];
rz(0.21342830511251873) q[3];
ry(-1.2009812992422897) q[4];
rz(1.8471552412133105) q[4];
ry(1.3384620176527715) q[5];
rz(0.5438453398086628) q[5];
ry(1.2540778069318037) q[6];
rz(-2.4086878766299824) q[6];
ry(-1.5542906457144046) q[7];
rz(3.097083611167536) q[7];
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
ry(-0.7006888531471471) q[0];
rz(-1.510602549322896) q[0];
ry(3.1306958049401117) q[1];
rz(1.5412421982656053) q[1];
ry(2.418647173809809) q[2];
rz(2.396426860127694) q[2];
ry(0.30425472118847213) q[3];
rz(3.134060107348535) q[3];
ry(-1.3329445668868916) q[4];
rz(-1.6848819688159475) q[4];
ry(-3.1341584295754483) q[5];
rz(-2.225634436205863) q[5];
ry(2.1296214006512173) q[6];
rz(0.1801702192115654) q[6];
ry(-0.6957626992548116) q[7];
rz(-0.18955368101932724) q[7];
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
ry(0.8603477419061106) q[0];
rz(2.60956590789268) q[0];
ry(-1.9685489057219527) q[1];
rz(2.882185291522939) q[1];
ry(0.4219609423736719) q[2];
rz(0.00024776551587013523) q[2];
ry(-1.535448559463655) q[3];
rz(-1.4588077063185843) q[3];
ry(-0.11639921003093592) q[4];
rz(1.845931993465808) q[4];
ry(1.3691183897958) q[5];
rz(-0.8333774528662605) q[5];
ry(-1.7407096261799921) q[6];
rz(3.0759470688423063) q[6];
ry(-3.071418888514879) q[7];
rz(-2.390259365932547) q[7];
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
ry(-1.6098593256441975) q[0];
rz(-0.3111662844282382) q[0];
ry(1.153441418159471) q[1];
rz(-0.28551947395012794) q[1];
ry(-0.6606732616527129) q[2];
rz(2.2335580719568813) q[2];
ry(1.0493421249525072) q[3];
rz(1.4538177894914435) q[3];
ry(0.28524495218619833) q[4];
rz(1.3010455472885478) q[4];
ry(1.6106937266264254) q[5];
rz(1.5670243413017342) q[5];
ry(0.3207927180586471) q[6];
rz(2.570762272390765) q[6];
ry(1.9709237785504667) q[7];
rz(1.9479157018181399) q[7];
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
ry(1.9018648699679703) q[0];
rz(1.502754030613783) q[0];
ry(-0.3387203002802798) q[1];
rz(-1.585827687947261) q[1];
ry(-2.5496869267) q[2];
rz(-2.1950800460498074) q[2];
ry(0.2641659381859185) q[3];
rz(2.842430689879417) q[3];
ry(2.3466755115011897) q[4];
rz(-0.7627724269851966) q[4];
ry(-0.3578539605909512) q[5];
rz(1.8205979150419378) q[5];
ry(-1.2658083332993275) q[6];
rz(2.7076960629151907) q[6];
ry(-2.8067636928989477) q[7];
rz(0.4417975408135666) q[7];
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
ry(2.67239847410916) q[0];
rz(-1.1482723246171487) q[0];
ry(-0.40938093877584475) q[1];
rz(-2.5390409925280366) q[1];
ry(0.6730136824799171) q[2];
rz(-0.1514807529906943) q[2];
ry(0.768098198188806) q[3];
rz(-1.0473102596846715) q[3];
ry(2.7692386567961953) q[4];
rz(0.2400702112866178) q[4];
ry(-2.3912059179228216) q[5];
rz(1.4742188295663397) q[5];
ry(1.334843448685949) q[6];
rz(0.6582634185133537) q[6];
ry(0.22097534124166707) q[7];
rz(1.5708903041747835) q[7];
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
ry(1.6493010931235625) q[0];
rz(-0.38277609730357026) q[0];
ry(1.5639487068043039) q[1];
rz(0.3254399187201633) q[1];
ry(-0.13030071426764422) q[2];
rz(2.7186159999627457) q[2];
ry(0.9552762517393615) q[3];
rz(-2.875382625576127) q[3];
ry(2.2564640189100444) q[4];
rz(-1.819521838817349) q[4];
ry(1.5051643864887652) q[5];
rz(1.2764705756182293) q[5];
ry(-1.9725224460380089) q[6];
rz(1.6758753823331674) q[6];
ry(-2.9359269967583304) q[7];
rz(-2.104144571915442) q[7];
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
ry(-0.5729253618390587) q[0];
rz(-2.802464828837304) q[0];
ry(-2.8243176734576934) q[1];
rz(-1.415662370847091) q[1];
ry(0.7803728137871433) q[2];
rz(1.6027011575931107) q[2];
ry(0.9172660668644597) q[3];
rz(-2.8850130996513306) q[3];
ry(-1.1063514446912608) q[4];
rz(-2.6561157364390837) q[4];
ry(-0.4772669108613288) q[5];
rz(2.229677526057717) q[5];
ry(-2.394258712789468) q[6];
rz(-2.0760286149199754) q[6];
ry(-1.09713258530088) q[7];
rz(1.0990746138556278) q[7];
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
ry(2.274081208928351) q[0];
rz(1.7045256895286718) q[0];
ry(1.9594720314230034) q[1];
rz(0.18041817823719697) q[1];
ry(-0.6504649691032613) q[2];
rz(0.3509304037264872) q[2];
ry(2.8714795086410025) q[3];
rz(0.10179584697146313) q[3];
ry(-2.402119910696284) q[4];
rz(1.3520081706057123) q[4];
ry(2.38717641610934) q[5];
rz(2.102027621391427) q[5];
ry(0.17941886504767535) q[6];
rz(-2.022711844281199) q[6];
ry(-1.233773213460477) q[7];
rz(-0.26166470879908466) q[7];
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
ry(-0.7009636896205572) q[0];
rz(1.6179422016480645) q[0];
ry(2.0587325241376067) q[1];
rz(2.2248350358430633) q[1];
ry(-2.4114021634443104) q[2];
rz(2.155545610473046) q[2];
ry(-2.884106293668917) q[3];
rz(0.09360042453763609) q[3];
ry(-0.2875491333543705) q[4];
rz(0.6813626004008361) q[4];
ry(3.076350246245786) q[5];
rz(-1.048458380560773) q[5];
ry(0.5547602374122966) q[6];
rz(2.607405708698689) q[6];
ry(2.580959857316803) q[7];
rz(-1.1448708370832037) q[7];
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
ry(1.6432132903150831) q[0];
rz(2.3717545635356156) q[0];
ry(1.8707068741682849) q[1];
rz(-0.14487541458789985) q[1];
ry(1.2178796784517214) q[2];
rz(-0.01656516449395628) q[2];
ry(-1.6504709641656312) q[3];
rz(1.256389425253356) q[3];
ry(2.635016889215064) q[4];
rz(-2.109328165453828) q[4];
ry(-0.8161250300814875) q[5];
rz(2.0413272348507077) q[5];
ry(-2.1155884105100116) q[6];
rz(-1.2153554000408588) q[6];
ry(1.9275103428484324) q[7];
rz(-1.6864334753886958) q[7];
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
ry(1.6504143922299797) q[0];
rz(-2.442576497481541) q[0];
ry(0.44644813199789457) q[1];
rz(-0.9235423112344163) q[1];
ry(2.6659920433993936) q[2];
rz(-2.626941995519744) q[2];
ry(-1.8962697297169362) q[3];
rz(0.4616762659165978) q[3];
ry(1.28676305126684) q[4];
rz(-0.46810293706599543) q[4];
ry(-2.7697635214637604) q[5];
rz(-1.6299076101515433) q[5];
ry(1.8551548273326375) q[6];
rz(2.971350679418493) q[6];
ry(1.960355393086389) q[7];
rz(-0.33933002827020614) q[7];
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
ry(1.8740334733894155) q[0];
rz(-1.315117478915108) q[0];
ry(1.0238605730394004) q[1];
rz(-0.15355614392485073) q[1];
ry(-1.723923593055828) q[2];
rz(0.1901600835097614) q[2];
ry(1.2667726566027877) q[3];
rz(-1.1844951031471078) q[3];
ry(-2.015326749907986) q[4];
rz(-0.9402913704160084) q[4];
ry(-0.6366953211144377) q[5];
rz(1.694528188763294) q[5];
ry(0.1449571579547797) q[6];
rz(0.646131747410438) q[6];
ry(2.6991156347183853) q[7];
rz(2.1685464331637645) q[7];
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
ry(0.9510681529542229) q[0];
rz(2.4856184695195847) q[0];
ry(2.906250096306473) q[1];
rz(2.6407904120579238) q[1];
ry(-0.3291023219839578) q[2];
rz(-2.9636442873590036) q[2];
ry(-0.48312549776635105) q[3];
rz(-1.4337214840240675) q[3];
ry(2.5748098034330154) q[4];
rz(-3.09574576050574) q[4];
ry(-1.3623527138650302) q[5];
rz(-0.9941124854693885) q[5];
ry(2.042739205674227) q[6];
rz(-0.3118292805765934) q[6];
ry(-2.4452300745115014) q[7];
rz(0.10108682145752501) q[7];
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
ry(-1.8507181215034039) q[0];
rz(-1.6259364263288092) q[0];
ry(1.2026070236029387) q[1];
rz(1.9896667119587201) q[1];
ry(2.5539859544223154) q[2];
rz(2.6777778598677364) q[2];
ry(-2.1821687778468357) q[3];
rz(3.0387480205042325) q[3];
ry(2.7877750167386726) q[4];
rz(-1.6193003352563358) q[4];
ry(-2.0385928911220805) q[5];
rz(-1.8761888899670343) q[5];
ry(1.810586077772598) q[6];
rz(-2.2015795634545396) q[6];
ry(-1.6015487296271067) q[7];
rz(2.9402308409896816) q[7];
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
ry(3.0684485859847355) q[0];
rz(1.038648925864548) q[0];
ry(-0.840218802686247) q[1];
rz(-1.085284428997091) q[1];
ry(0.5387493956563363) q[2];
rz(-1.985447814828914) q[2];
ry(-0.916778028292919) q[3];
rz(-2.5817494693356102) q[3];
ry(-2.9884226429825147) q[4];
rz(2.3516161035635057) q[4];
ry(-0.4820339842781314) q[5];
rz(-1.52224574489298) q[5];
ry(-0.6638924355260157) q[6];
rz(0.33900646096373693) q[6];
ry(2.28465247037131) q[7];
rz(-1.8417879380040976) q[7];
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
ry(3.0397460143974873) q[0];
rz(0.9129089806049335) q[0];
ry(0.011539256249249532) q[1];
rz(0.16152934471324354) q[1];
ry(-2.3006473902447717) q[2];
rz(0.11888607922505745) q[2];
ry(0.7198966450339581) q[3];
rz(0.4676067608683446) q[3];
ry(-1.9119212338076514) q[4];
rz(-2.5141080417551604) q[4];
ry(-1.7809254931498966) q[5];
rz(-0.8059972056518541) q[5];
ry(0.3683090492636678) q[6];
rz(1.556167856705703) q[6];
ry(-1.3979655588417916) q[7];
rz(-1.031211131271867) q[7];
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
ry(2.581861436493014) q[0];
rz(1.2280391225328193) q[0];
ry(-0.15683001919849318) q[1];
rz(-0.44846474616315835) q[1];
ry(2.013227271900584) q[2];
rz(2.1212820584475613) q[2];
ry(-1.187902657897168) q[3];
rz(-1.6921266524579774) q[3];
ry(-0.8838265029697326) q[4];
rz(-1.88808591683228) q[4];
ry(2.4115050589443157) q[5];
rz(-1.506168046057921) q[5];
ry(3.0309954993734762) q[6];
rz(2.3702659101546004) q[6];
ry(-0.08801489238412638) q[7];
rz(-1.0544248167813082) q[7];