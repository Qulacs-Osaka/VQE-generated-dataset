OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.3462190906517) q[0];
ry(-1.9945981676398625) q[1];
cx q[0],q[1];
ry(2.9716856744738513) q[0];
ry(-1.927602574662905) q[1];
cx q[0],q[1];
ry(0.5077713865545116) q[2];
ry(2.560153411577146) q[3];
cx q[2],q[3];
ry(2.484665153708997) q[2];
ry(-1.3325461889192933) q[3];
cx q[2],q[3];
ry(-0.5055512731631353) q[0];
ry(0.49413096941065265) q[2];
cx q[0],q[2];
ry(1.2065984196524606) q[0];
ry(-0.07181135367873653) q[2];
cx q[0],q[2];
ry(0.848566427001728) q[1];
ry(-0.5655728485339456) q[3];
cx q[1],q[3];
ry(-2.9173296426104707) q[1];
ry(1.4172043509086596) q[3];
cx q[1],q[3];
ry(2.0324160256907606) q[0];
ry(0.8086890259001001) q[3];
cx q[0],q[3];
ry(-1.3025418065056407) q[0];
ry(1.774796681543061) q[3];
cx q[0],q[3];
ry(-1.5189838303337293) q[1];
ry(0.27380829917164395) q[2];
cx q[1],q[2];
ry(1.7520362846370314) q[1];
ry(0.42700577448307947) q[2];
cx q[1],q[2];
ry(-0.15932090011140654) q[0];
ry(1.1174065076665727) q[1];
cx q[0],q[1];
ry(-0.9497895966499472) q[0];
ry(-0.32832314937488416) q[1];
cx q[0],q[1];
ry(0.6501364384094254) q[2];
ry(-1.5819114481168557) q[3];
cx q[2],q[3];
ry(2.2972039123075136) q[2];
ry(2.3743826061074285) q[3];
cx q[2],q[3];
ry(-2.3579576334250354) q[0];
ry(0.48141673357946724) q[2];
cx q[0],q[2];
ry(2.089333246588679) q[0];
ry(-1.4762722868677844) q[2];
cx q[0],q[2];
ry(-1.9177674864382546) q[1];
ry(-1.6244916274368124) q[3];
cx q[1],q[3];
ry(0.8601063842521555) q[1];
ry(1.6806911152693962) q[3];
cx q[1],q[3];
ry(-0.8599304964928383) q[0];
ry(0.41892011712112964) q[3];
cx q[0],q[3];
ry(-2.934144679211187) q[0];
ry(-2.3315337105410254) q[3];
cx q[0],q[3];
ry(-0.4879093398514441) q[1];
ry(-0.46748577638823813) q[2];
cx q[1],q[2];
ry(-1.3002962722481524) q[1];
ry(-2.0550194016593184) q[2];
cx q[1],q[2];
ry(2.2842615580933634) q[0];
ry(0.524118764010816) q[1];
cx q[0],q[1];
ry(-2.9326776097659026) q[0];
ry(0.10840374109346484) q[1];
cx q[0],q[1];
ry(-1.6520193725224377) q[2];
ry(2.406677506033391) q[3];
cx q[2],q[3];
ry(-2.3146045733014) q[2];
ry(-0.5528866153118926) q[3];
cx q[2],q[3];
ry(-0.1766978784924057) q[0];
ry(-2.7529495980842666) q[2];
cx q[0],q[2];
ry(-0.023922548674256517) q[0];
ry(-0.1787822896587081) q[2];
cx q[0],q[2];
ry(-2.82136048214903) q[1];
ry(-1.8372810054197648) q[3];
cx q[1],q[3];
ry(0.5442171676930669) q[1];
ry(1.2358314820411893) q[3];
cx q[1],q[3];
ry(1.8590097923139852) q[0];
ry(-1.2257243041981543) q[3];
cx q[0],q[3];
ry(2.6657688462869755) q[0];
ry(-1.289548780131493) q[3];
cx q[0],q[3];
ry(1.8757114602808578) q[1];
ry(-2.4914236523872892) q[2];
cx q[1],q[2];
ry(3.04498743286445) q[1];
ry(3.0584094424000194) q[2];
cx q[1],q[2];
ry(1.5787515980884255) q[0];
ry(-0.6440509108157757) q[1];
cx q[0],q[1];
ry(0.9129791621234334) q[0];
ry(3.0797242616503557) q[1];
cx q[0],q[1];
ry(-1.188798624939893) q[2];
ry(-2.1769877607624855) q[3];
cx q[2],q[3];
ry(1.9820117966717392) q[2];
ry(2.1827664992486806) q[3];
cx q[2],q[3];
ry(-0.20380036674807925) q[0];
ry(1.4653312524478412) q[2];
cx q[0],q[2];
ry(-0.9214985469137632) q[0];
ry(0.7680049089983393) q[2];
cx q[0],q[2];
ry(1.7552800174950716) q[1];
ry(2.0438463265284037) q[3];
cx q[1],q[3];
ry(2.6951128696307536) q[1];
ry(0.11222146376292665) q[3];
cx q[1],q[3];
ry(1.8815616607286263) q[0];
ry(-0.15694465077996417) q[3];
cx q[0],q[3];
ry(-0.4242056541044157) q[0];
ry(0.3422767030456253) q[3];
cx q[0],q[3];
ry(-2.1889062799837307) q[1];
ry(0.6577518205113222) q[2];
cx q[1],q[2];
ry(2.097119169720359) q[1];
ry(0.593855780582252) q[2];
cx q[1],q[2];
ry(-2.2085586610738672) q[0];
ry(-1.7983207199473297) q[1];
cx q[0],q[1];
ry(-2.932110329396229) q[0];
ry(1.8942916739933633) q[1];
cx q[0],q[1];
ry(1.5506225491899421) q[2];
ry(-2.0887059084503803) q[3];
cx q[2],q[3];
ry(1.769441655539775) q[2];
ry(-1.8979542039749357) q[3];
cx q[2],q[3];
ry(2.997895831386954) q[0];
ry(-1.4579326635962249) q[2];
cx q[0],q[2];
ry(-3.093788417646274) q[0];
ry(1.0145182804360977) q[2];
cx q[0],q[2];
ry(1.1717828481183148) q[1];
ry(-2.2331146890571256) q[3];
cx q[1],q[3];
ry(-1.3274190381859814) q[1];
ry(1.5868833382660474) q[3];
cx q[1],q[3];
ry(-1.1621710742346132) q[0];
ry(-2.658653565239841) q[3];
cx q[0],q[3];
ry(0.5771032372196303) q[0];
ry(-1.0482302449760854) q[3];
cx q[0],q[3];
ry(-0.45563804284958387) q[1];
ry(-0.004912077751204613) q[2];
cx q[1],q[2];
ry(1.4586426552961163) q[1];
ry(2.5144370614152898) q[2];
cx q[1],q[2];
ry(-2.5457326274866454) q[0];
ry(-1.3897275966008467) q[1];
cx q[0],q[1];
ry(-3.1168094062537466) q[0];
ry(1.3665508037917904) q[1];
cx q[0],q[1];
ry(-2.241056511554878) q[2];
ry(-1.8265151609916372) q[3];
cx q[2],q[3];
ry(1.6492000343741753) q[2];
ry(-1.1324739973370752) q[3];
cx q[2],q[3];
ry(-1.5236005470956415) q[0];
ry(-0.9470816855975759) q[2];
cx q[0],q[2];
ry(2.691737989553877) q[0];
ry(0.6989546261824573) q[2];
cx q[0],q[2];
ry(2.9724290022644086) q[1];
ry(2.0156630814335377) q[3];
cx q[1],q[3];
ry(-2.8805148799481786) q[1];
ry(1.4800510498092843) q[3];
cx q[1],q[3];
ry(1.5410169438015007) q[0];
ry(-0.47685957411017377) q[3];
cx q[0],q[3];
ry(-0.24485053079905764) q[0];
ry(0.7260927660416465) q[3];
cx q[0],q[3];
ry(-0.19032812652671804) q[1];
ry(-2.7182940355635763) q[2];
cx q[1],q[2];
ry(2.1895253249363424) q[1];
ry(1.419649015339992) q[2];
cx q[1],q[2];
ry(-0.27665019667984403) q[0];
ry(-2.2523275291452682) q[1];
cx q[0],q[1];
ry(-2.7671782510637657) q[0];
ry(3.108928315964384) q[1];
cx q[0],q[1];
ry(-2.60288270625975) q[2];
ry(-0.1469760090434127) q[3];
cx q[2],q[3];
ry(3.029349836689951) q[2];
ry(1.216799671336954) q[3];
cx q[2],q[3];
ry(2.6627530537499164) q[0];
ry(-3.1283137802613736) q[2];
cx q[0],q[2];
ry(-1.8973073074718478) q[0];
ry(-1.585895232230773) q[2];
cx q[0],q[2];
ry(2.320116138474666) q[1];
ry(-2.8002562972124685) q[3];
cx q[1],q[3];
ry(0.1770629382796205) q[1];
ry(2.8572087266228468) q[3];
cx q[1],q[3];
ry(-0.29675817203962507) q[0];
ry(-1.4349735531351682) q[3];
cx q[0],q[3];
ry(0.539162188674949) q[0];
ry(-1.2110501868602137) q[3];
cx q[0],q[3];
ry(-0.9601108319367748) q[1];
ry(-2.4908096376519393) q[2];
cx q[1],q[2];
ry(-1.0372217474550158) q[1];
ry(1.1920289882837691) q[2];
cx q[1],q[2];
ry(-3.0000272194287247) q[0];
ry(3.0285512398544827) q[1];
cx q[0],q[1];
ry(2.6082385864189335) q[0];
ry(-1.4685150830256128) q[1];
cx q[0],q[1];
ry(-1.6258288966685661) q[2];
ry(-1.5661782454398754) q[3];
cx q[2],q[3];
ry(-1.65374382292607) q[2];
ry(0.14794324133315567) q[3];
cx q[2],q[3];
ry(2.0799272172501055) q[0];
ry(2.9473753505125524) q[2];
cx q[0],q[2];
ry(1.571989370687719) q[0];
ry(0.450868106251475) q[2];
cx q[0],q[2];
ry(0.29301213241831725) q[1];
ry(-0.8188485611138018) q[3];
cx q[1],q[3];
ry(2.850363131408137) q[1];
ry(-0.5500621232684292) q[3];
cx q[1],q[3];
ry(-1.4290116995249424) q[0];
ry(2.279615412870367) q[3];
cx q[0],q[3];
ry(0.03315751028606793) q[0];
ry(-0.5802165368736437) q[3];
cx q[0],q[3];
ry(-0.009141330911374368) q[1];
ry(-0.9675245029494773) q[2];
cx q[1],q[2];
ry(2.389769364868425) q[1];
ry(-2.1409952977827285) q[2];
cx q[1],q[2];
ry(-0.8093075571805554) q[0];
ry(1.598510095612889) q[1];
cx q[0],q[1];
ry(-2.1343968145997128) q[0];
ry(3.0128752532395824) q[1];
cx q[0],q[1];
ry(0.39400497688364583) q[2];
ry(1.030026988624221) q[3];
cx q[2],q[3];
ry(1.7052149399480396) q[2];
ry(0.433962565428251) q[3];
cx q[2],q[3];
ry(-2.4416698325530213) q[0];
ry(1.5920396573052802) q[2];
cx q[0],q[2];
ry(1.7347486080654404) q[0];
ry(2.067873422890794) q[2];
cx q[0],q[2];
ry(2.3791797147836364) q[1];
ry(-2.2359166565591115) q[3];
cx q[1],q[3];
ry(-0.65922134446038) q[1];
ry(0.5898854032834965) q[3];
cx q[1],q[3];
ry(0.9239264540759337) q[0];
ry(-0.0035332746072298652) q[3];
cx q[0],q[3];
ry(-1.7569517942813988) q[0];
ry(-1.698166324054566) q[3];
cx q[0],q[3];
ry(-2.7811564565690983) q[1];
ry(-2.80231980844083) q[2];
cx q[1],q[2];
ry(-1.7428401976600263) q[1];
ry(-0.6276566894757798) q[2];
cx q[1],q[2];
ry(-0.556153207029619) q[0];
ry(-0.4933448827770102) q[1];
cx q[0],q[1];
ry(1.762278154053596) q[0];
ry(0.44927669885887267) q[1];
cx q[0],q[1];
ry(1.1321902873387106) q[2];
ry(-0.4451041560278144) q[3];
cx q[2],q[3];
ry(1.2530012144587321) q[2];
ry(0.6087703362669166) q[3];
cx q[2],q[3];
ry(0.6566256883437258) q[0];
ry(-1.486080040632458) q[2];
cx q[0],q[2];
ry(0.5476744514560838) q[0];
ry(-0.17792988765061946) q[2];
cx q[0],q[2];
ry(-2.6091180056764736) q[1];
ry(1.5908913802991145) q[3];
cx q[1],q[3];
ry(-1.8116336718701391) q[1];
ry(1.8000214141287314) q[3];
cx q[1],q[3];
ry(-1.110761941652088) q[0];
ry(2.4254747773858396) q[3];
cx q[0],q[3];
ry(0.07541866136488744) q[0];
ry(2.86397319518503) q[3];
cx q[0],q[3];
ry(-0.30174831101230204) q[1];
ry(0.4577289197535901) q[2];
cx q[1],q[2];
ry(2.8667972475110868) q[1];
ry(0.5998455610351164) q[2];
cx q[1],q[2];
ry(2.099287846484258) q[0];
ry(0.44317569406752805) q[1];
cx q[0],q[1];
ry(2.293525033853987) q[0];
ry(-2.3967367134019235) q[1];
cx q[0],q[1];
ry(0.6130323660899175) q[2];
ry(1.2605493833794468) q[3];
cx q[2],q[3];
ry(-2.448537831272113) q[2];
ry(-1.0007637066215178) q[3];
cx q[2],q[3];
ry(1.804327251621011) q[0];
ry(-1.8415803348147555) q[2];
cx q[0],q[2];
ry(-0.04993578850063721) q[0];
ry(-1.0980741563170706) q[2];
cx q[0],q[2];
ry(-1.9738118962015472) q[1];
ry(-3.0013287031760143) q[3];
cx q[1],q[3];
ry(-1.9344666222110858) q[1];
ry(1.1807497410051457) q[3];
cx q[1],q[3];
ry(-2.9876364870718564) q[0];
ry(2.6679615135245185) q[3];
cx q[0],q[3];
ry(2.9409431021871897) q[0];
ry(0.85908395947007) q[3];
cx q[0],q[3];
ry(0.3495697058784133) q[1];
ry(-2.6784102793724536) q[2];
cx q[1],q[2];
ry(-2.7180203119731874) q[1];
ry(-1.425246657423803) q[2];
cx q[1],q[2];
ry(2.534476279919213) q[0];
ry(0.24525419928082837) q[1];
cx q[0],q[1];
ry(-3.070365271524544) q[0];
ry(-2.111209924058631) q[1];
cx q[0],q[1];
ry(-2.923382132834951) q[2];
ry(-0.3161114590380727) q[3];
cx q[2],q[3];
ry(-1.9070976882583939) q[2];
ry(-1.0400469280763962) q[3];
cx q[2],q[3];
ry(3.0227240455245328) q[0];
ry(1.1954651292062168) q[2];
cx q[0],q[2];
ry(1.6475443548114452) q[0];
ry(2.276812697611649) q[2];
cx q[0],q[2];
ry(-1.8168715117350698) q[1];
ry(-2.0394996965101546) q[3];
cx q[1],q[3];
ry(0.7673762698441281) q[1];
ry(1.124056639391604) q[3];
cx q[1],q[3];
ry(2.276368883309801) q[0];
ry(-0.5596752703792829) q[3];
cx q[0],q[3];
ry(0.6952936182082506) q[0];
ry(-2.44703050700298) q[3];
cx q[0],q[3];
ry(-2.6967267325750384) q[1];
ry(-1.4812586742060951) q[2];
cx q[1],q[2];
ry(-2.2681807791605237) q[1];
ry(1.143738438677273) q[2];
cx q[1],q[2];
ry(0.001079075648486351) q[0];
ry(1.2082379995253594) q[1];
cx q[0],q[1];
ry(2.4580374575812254) q[0];
ry(2.3950611197991294) q[1];
cx q[0],q[1];
ry(0.4046943574931652) q[2];
ry(2.1006460481934206) q[3];
cx q[2],q[3];
ry(1.8667753184601232) q[2];
ry(2.280100372425321) q[3];
cx q[2],q[3];
ry(0.20481717095288032) q[0];
ry(-2.357906415956711) q[2];
cx q[0],q[2];
ry(-0.8411043058700498) q[0];
ry(1.0197560134020618) q[2];
cx q[0],q[2];
ry(0.27458686951581535) q[1];
ry(2.216327380297866) q[3];
cx q[1],q[3];
ry(-1.4718907229725895) q[1];
ry(-0.26206265624142855) q[3];
cx q[1],q[3];
ry(-1.9649931441996031) q[0];
ry(2.2618695288256303) q[3];
cx q[0],q[3];
ry(-3.0729769540311733) q[0];
ry(-2.7276329342122043) q[3];
cx q[0],q[3];
ry(-0.791485157391455) q[1];
ry(2.4581029237534464) q[2];
cx q[1],q[2];
ry(1.2379203825991891) q[1];
ry(1.3908816139882518) q[2];
cx q[1],q[2];
ry(1.8198960981511052) q[0];
ry(-1.3507620433389234) q[1];
cx q[0],q[1];
ry(-1.072419842614611) q[0];
ry(-0.7672710388133447) q[1];
cx q[0],q[1];
ry(-1.3745460371266187) q[2];
ry(1.5030317518675835) q[3];
cx q[2],q[3];
ry(2.5585570689172727) q[2];
ry(2.97687259201722) q[3];
cx q[2],q[3];
ry(0.27952754252954753) q[0];
ry(2.1250066395169154) q[2];
cx q[0],q[2];
ry(2.0407496280674877) q[0];
ry(1.9167111294867976) q[2];
cx q[0],q[2];
ry(-2.149923786638044) q[1];
ry(2.159161613751731) q[3];
cx q[1],q[3];
ry(-1.9801019422576185) q[1];
ry(-0.5127877114379392) q[3];
cx q[1],q[3];
ry(0.935956731532011) q[0];
ry(0.18117888381437086) q[3];
cx q[0],q[3];
ry(-1.3360755619229516) q[0];
ry(1.3597978792032026) q[3];
cx q[0],q[3];
ry(1.2231810509445573) q[1];
ry(-0.11774052111813305) q[2];
cx q[1],q[2];
ry(-2.1701211717254187) q[1];
ry(-2.9633932174489854) q[2];
cx q[1],q[2];
ry(-0.1429790994383513) q[0];
ry(-0.37373657522497794) q[1];
cx q[0],q[1];
ry(1.3068031053038316) q[0];
ry(2.6543760151682805) q[1];
cx q[0],q[1];
ry(-1.6828210912215322) q[2];
ry(1.7922094335550467) q[3];
cx q[2],q[3];
ry(-2.0174045715890543) q[2];
ry(0.5200897127585807) q[3];
cx q[2],q[3];
ry(2.262019321925356) q[0];
ry(-1.1885266481714076) q[2];
cx q[0],q[2];
ry(1.617861101670757) q[0];
ry(-0.8328925741907964) q[2];
cx q[0],q[2];
ry(-2.3745557096669847) q[1];
ry(-1.649142055477216) q[3];
cx q[1],q[3];
ry(0.8155741032644048) q[1];
ry(1.785009373655043) q[3];
cx q[1],q[3];
ry(-2.3354951691943286) q[0];
ry(0.429842606540305) q[3];
cx q[0],q[3];
ry(-0.626075637156043) q[0];
ry(2.238535174621491) q[3];
cx q[0],q[3];
ry(2.516435084980217) q[1];
ry(-1.487560572814246) q[2];
cx q[1],q[2];
ry(0.39691622625184664) q[1];
ry(-0.16328056562244345) q[2];
cx q[1],q[2];
ry(-1.396600691162253) q[0];
ry(2.2806869757005597) q[1];
cx q[0],q[1];
ry(-1.6953429693868665) q[0];
ry(1.341358844702028) q[1];
cx q[0],q[1];
ry(-1.7178748835082605) q[2];
ry(3.078539102519975) q[3];
cx q[2],q[3];
ry(0.35646947770302667) q[2];
ry(1.3255755980572077) q[3];
cx q[2],q[3];
ry(2.992105878100108) q[0];
ry(0.3205481369623602) q[2];
cx q[0],q[2];
ry(1.7012682565889392) q[0];
ry(-2.7267602442101917) q[2];
cx q[0],q[2];
ry(-0.21552844788250436) q[1];
ry(-2.162176491550367) q[3];
cx q[1],q[3];
ry(1.9270061356287962) q[1];
ry(-0.3509938811510125) q[3];
cx q[1],q[3];
ry(-2.7065061302557822) q[0];
ry(-1.9707798185609657) q[3];
cx q[0],q[3];
ry(-0.3761560496284515) q[0];
ry(-0.39836234296440054) q[3];
cx q[0],q[3];
ry(2.172944948062466) q[1];
ry(-1.7056113279013765) q[2];
cx q[1],q[2];
ry(-1.3494108386050236) q[1];
ry(2.878409063305839) q[2];
cx q[1],q[2];
ry(-1.6783368038367952) q[0];
ry(-0.013972783617263174) q[1];
cx q[0],q[1];
ry(-2.2064013502799886) q[0];
ry(-1.342909390760406) q[1];
cx q[0],q[1];
ry(-1.9298962180697314) q[2];
ry(0.41343793574100796) q[3];
cx q[2],q[3];
ry(3.0471593641888575) q[2];
ry(-1.8960797831010572) q[3];
cx q[2],q[3];
ry(-2.7173443381820053) q[0];
ry(2.074074764025667) q[2];
cx q[0],q[2];
ry(-1.3325904071347023) q[0];
ry(1.0612791316424606) q[2];
cx q[0],q[2];
ry(2.0400171703325705) q[1];
ry(2.0344059557013634) q[3];
cx q[1],q[3];
ry(-2.569226269105706) q[1];
ry(0.28774263319880067) q[3];
cx q[1],q[3];
ry(0.27083230398698444) q[0];
ry(-1.9039662483380608) q[3];
cx q[0],q[3];
ry(3.0364556153192654) q[0];
ry(0.6390539282584865) q[3];
cx q[0],q[3];
ry(-2.094660059896123) q[1];
ry(2.445368182527618) q[2];
cx q[1],q[2];
ry(1.5449602919441292) q[1];
ry(-1.0104165052256997) q[2];
cx q[1],q[2];
ry(1.4543154805952578) q[0];
ry(1.5862638597105017) q[1];
cx q[0],q[1];
ry(-1.9686675465884917) q[0];
ry(0.6314237888743198) q[1];
cx q[0],q[1];
ry(-0.30572734657033585) q[2];
ry(2.3123200943489017) q[3];
cx q[2],q[3];
ry(-0.9981659219365032) q[2];
ry(-0.10746409642496331) q[3];
cx q[2],q[3];
ry(-2.6945018755254493) q[0];
ry(3.0691913790063374) q[2];
cx q[0],q[2];
ry(0.40308210140830647) q[0];
ry(0.9598346878631296) q[2];
cx q[0],q[2];
ry(-0.9824482696113526) q[1];
ry(-0.4864011243694) q[3];
cx q[1],q[3];
ry(-1.290944418433998) q[1];
ry(2.089599226234823) q[3];
cx q[1],q[3];
ry(1.6023942058777472) q[0];
ry(2.299604296269736) q[3];
cx q[0],q[3];
ry(-2.571627681856197) q[0];
ry(2.6270216688035437) q[3];
cx q[0],q[3];
ry(0.5928715572245487) q[1];
ry(-2.513573791223334) q[2];
cx q[1],q[2];
ry(2.099406800939912) q[1];
ry(-0.7274929699447243) q[2];
cx q[1],q[2];
ry(-2.3411491517846024) q[0];
ry(-0.2499566228569969) q[1];
cx q[0],q[1];
ry(-1.3067489202605755) q[0];
ry(0.5111451005682482) q[1];
cx q[0],q[1];
ry(0.9505018149544507) q[2];
ry(0.0689552397680022) q[3];
cx q[2],q[3];
ry(-0.7339143047700576) q[2];
ry(0.5420070293159599) q[3];
cx q[2],q[3];
ry(2.398258964863722) q[0];
ry(-2.678251043309198) q[2];
cx q[0],q[2];
ry(-0.6218556011363063) q[0];
ry(0.540131523971743) q[2];
cx q[0],q[2];
ry(-2.145364948074983) q[1];
ry(-0.8546883562712125) q[3];
cx q[1],q[3];
ry(3.029953399102794) q[1];
ry(-3.1377919028154917) q[3];
cx q[1],q[3];
ry(-2.683116140938924) q[0];
ry(1.8450493959805607) q[3];
cx q[0],q[3];
ry(-3.0750517133441337) q[0];
ry(-2.0333918740505297) q[3];
cx q[0],q[3];
ry(-1.3672405829190444) q[1];
ry(0.5745139688738456) q[2];
cx q[1],q[2];
ry(-3.0571618336594923) q[1];
ry(0.6334356215951816) q[2];
cx q[1],q[2];
ry(-0.6752951781252797) q[0];
ry(0.45527201592424493) q[1];
cx q[0],q[1];
ry(-2.941686473179287) q[0];
ry(-2.9388393112084406) q[1];
cx q[0],q[1];
ry(2.7244322733440116) q[2];
ry(-2.3733878773826658) q[3];
cx q[2],q[3];
ry(-0.856994135436187) q[2];
ry(1.0947701642011598) q[3];
cx q[2],q[3];
ry(-0.25085669686899903) q[0];
ry(2.918351396334101) q[2];
cx q[0],q[2];
ry(-2.429632902464646) q[0];
ry(-1.9186242091446022) q[2];
cx q[0],q[2];
ry(-0.6959419133800803) q[1];
ry(-1.6971099125887001) q[3];
cx q[1],q[3];
ry(-3.003692360042774) q[1];
ry(3.0941994169721863) q[3];
cx q[1],q[3];
ry(2.7655752251817622) q[0];
ry(-1.6332733890711317) q[3];
cx q[0],q[3];
ry(1.6034134563860865) q[0];
ry(-2.536421736651773) q[3];
cx q[0],q[3];
ry(0.5867679709831704) q[1];
ry(2.050093932886445) q[2];
cx q[1],q[2];
ry(-2.991920284648408) q[1];
ry(0.0600557155947925) q[2];
cx q[1],q[2];
ry(-1.3303227710505363) q[0];
ry(-2.515871605325075) q[1];
ry(1.6006762926031834) q[2];
ry(1.359644887048331) q[3];