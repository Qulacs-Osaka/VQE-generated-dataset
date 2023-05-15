OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.9333035607396356) q[0];
rz(0.23787682607351596) q[0];
ry(3.0179971277997666) q[1];
rz(1.095338156482324) q[1];
ry(-0.24785817522861908) q[2];
rz(3.0095947466060884) q[2];
ry(3.1366809603535843) q[3];
rz(0.4426808864735392) q[3];
ry(2.9367853858449258) q[4];
rz(0.08609188730090445) q[4];
ry(-2.9224495857200083) q[5];
rz(1.1571152126103488) q[5];
ry(-0.9468012247046831) q[6];
rz(-1.5595028127982846) q[6];
ry(-1.5439114504457374) q[7];
rz(-0.9515157411872703) q[7];
ry(-0.786883064795888) q[8];
rz(0.053008962396184245) q[8];
ry(3.1415543266022863) q[9];
rz(-0.5825296850210278) q[9];
ry(3.09832722173261) q[10];
rz(-0.446094696625509) q[10];
ry(3.108302458671812) q[11];
rz(0.15370960272963516) q[11];
ry(0.0010106122098179071) q[12];
rz(2.5281741474135964) q[12];
ry(0.2297820738583347) q[13];
rz(-2.99781926099161) q[13];
ry(2.609067166625596) q[14];
rz(3.0664664718844614) q[14];
ry(3.1407142264935732) q[15];
rz(0.9105248540747093) q[15];
ry(3.134812670851379) q[16];
rz(-1.9033730630816623) q[16];
ry(-3.0708294669827483) q[17];
rz(-1.1266359210739478) q[17];
ry(-0.12035294245929151) q[18];
rz(0.09684971471022408) q[18];
ry(0.7553564806751911) q[19];
rz(-0.4632380264405658) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.3630008052980936) q[0];
rz(-1.6437972411640305) q[0];
ry(-0.583646801712141) q[1];
rz(-1.2504890277666112) q[1];
ry(2.9312815233492646) q[2];
rz(1.2290010152855562) q[2];
ry(-0.025827935671319125) q[3];
rz(0.4545028355614257) q[3];
ry(-2.913519335851921) q[4];
rz(0.6994356174129147) q[4];
ry(0.00023352437880526096) q[5];
rz(0.3953884547164233) q[5];
ry(-3.1398633327479515) q[6];
rz(-1.72665240520318) q[6];
ry(3.030252956650408) q[7];
rz(-1.4831256030205981) q[7];
ry(1.543979114759374) q[8];
rz(-3.009134802184685) q[8];
ry(-3.2011355667515177e-05) q[9];
rz(1.7906862290626027) q[9];
ry(2.6601798156539385) q[10];
rz(2.9882841883319258) q[10];
ry(-0.05378572790569436) q[11];
rz(2.5623560087505095) q[11];
ry(-3.1358007305986084) q[12];
rz(0.8503920775887818) q[12];
ry(0.3482338967549935) q[13];
rz(0.6818228754302389) q[13];
ry(0.5760152819555309) q[14];
rz(-2.636346813209173) q[14];
ry(-0.032704899827637135) q[15];
rz(0.8144443196947201) q[15];
ry(-3.1123398100566804) q[16];
rz(-2.577986929415857) q[16];
ry(3.0715783446304132) q[17];
rz(-2.9608496773852817) q[17];
ry(2.4926852739562873) q[18];
rz(-0.37800355209445424) q[18];
ry(-0.7619804535594036) q[19];
rz(0.5067264708237142) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.2532282990846939) q[0];
rz(-0.45395765803086696) q[0];
ry(-3.06884301668406) q[1];
rz(0.63765628641094) q[1];
ry(-2.6982445999847715) q[2];
rz(2.855983282445664) q[2];
ry(-0.002827949639132577) q[3];
rz(0.3572770699836907) q[3];
ry(3.1274971171926538) q[4];
rz(-1.0651362820008357) q[4];
ry(-1.355983321252761) q[5];
rz(-2.9242809996891452) q[5];
ry(-2.538367357499627) q[6];
rz(-2.6973116949295908) q[6];
ry(-2.4005121682924946) q[7];
rz(0.6580418975769744) q[7];
ry(1.5538692215695997) q[8];
rz(-2.505334070756197) q[8];
ry(1.5707844071519792) q[9];
rz(0.6298445824729582) q[9];
ry(1.9531381886933392) q[10];
rz(-3.0492350752783652) q[10];
ry(-0.005000210526365123) q[11];
rz(-1.6772800595581492) q[11];
ry(1.435076388027583) q[12];
rz(-2.795542041097805) q[12];
ry(1.5904928724818603) q[13];
rz(0.5885556522997961) q[13];
ry(-3.1117985007215143) q[14];
rz(-0.0667306501380924) q[14];
ry(3.132974303871332) q[15];
rz(-2.080156307548082) q[15];
ry(3.1249400987316656) q[16];
rz(2.4691961256926844) q[16];
ry(0.011729307167344771) q[17];
rz(0.2868677760234551) q[17];
ry(-0.12760400184370102) q[18];
rz(-2.6556992363040877) q[18];
ry(2.9677667517878143) q[19];
rz(-0.8927213669629531) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.4216398464716233) q[0];
rz(-2.82870425408359) q[0];
ry(-3.1084530157370156) q[1];
rz(-2.6607797665866952) q[1];
ry(1.2341222867513504) q[2];
rz(-0.09388979936145248) q[2];
ry(-3.0602424959935517) q[3];
rz(2.6625943984270206) q[3];
ry(-3.0853251971018705) q[4];
rz(-2.9116811677536427) q[4];
ry(-1.4545962504098835) q[5];
rz(0.044885649004842) q[5];
ry(-3.1412083818835974) q[6];
rz(1.8740087719394136) q[6];
ry(-3.1139312525419798) q[7];
rz(-1.447814357099492) q[7];
ry(-3.137604339735657) q[8];
rz(0.8073338653760852) q[8];
ry(0.031109340227357514) q[9];
rz(-1.652925131778809) q[9];
ry(-1.5708085725935632) q[10];
rz(2.8941246185930924) q[10];
ry(3.532765330671595e-05) q[11];
rz(2.3276666007095455) q[11];
ry(-3.124289553486101) q[12];
rz(0.3486192727238695) q[12];
ry(0.008341451260786257) q[13];
rz(2.724372576661215) q[13];
ry(-1.5876037627736321) q[14];
rz(2.12329512666178) q[14];
ry(3.1140651881029138) q[15];
rz(-1.3657609017740868) q[15];
ry(2.3248402610882732) q[16];
rz(2.4646812932295274) q[16];
ry(-1.2481516275937898) q[17];
rz(-1.9636512537936301) q[17];
ry(1.7587686748612141) q[18];
rz(0.3422752687740743) q[18];
ry(-2.4334580189118085) q[19];
rz(2.2534766903947387) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.727424344412913) q[0];
rz(-1.3692223866742816) q[0];
ry(3.0847536091379997) q[1];
rz(1.6468842124984815) q[1];
ry(-0.021880437517783236) q[2];
rz(-2.5062470283893026) q[2];
ry(0.004610041865898751) q[3];
rz(0.15718490421980835) q[3];
ry(-2.9563842629672203) q[4];
rz(-1.0820038992580903) q[4];
ry(-1.3563920529111275) q[5];
rz(-1.5607625455464698) q[5];
ry(-0.013361808871686698) q[6];
rz(-3.1189988094492995) q[6];
ry(1.1429369712460549) q[7];
rz(3.008745196070752) q[7];
ry(-2.938915308162704) q[8];
rz(-3.110461761184486) q[8];
ry(0.45163653189609415) q[9];
rz(-2.1209799427636264) q[9];
ry(2.297980490293214) q[10];
rz(2.0100864490975496) q[10];
ry(1.5707846881580192) q[11];
rz(0.6927661452027893) q[11];
ry(1.4281122463271079) q[12];
rz(0.2459066981465935) q[12];
ry(-0.17359920965611242) q[13];
rz(-0.024291062032916873) q[13];
ry(-0.588740058574235) q[14];
rz(1.316098639852616) q[14];
ry(-3.061081967076204) q[15];
rz(0.9095525817078585) q[15];
ry(-0.0026929623495204804) q[16];
rz(-2.6050118876350288) q[16];
ry(-3.133940389759927) q[17];
rz(-1.8589466538441874) q[17];
ry(0.009786804566850547) q[18];
rz(-2.9286080193686104) q[18];
ry(-2.565173075433271) q[19];
rz(0.9916142412933554) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.8154362768889454) q[0];
rz(-2.8600860767663523) q[0];
ry(-2.399178330436507) q[1];
rz(-2.9269153248383257) q[1];
ry(1.2648888507502218) q[2];
rz(2.7397174927595866) q[2];
ry(-0.07363508836209043) q[3];
rz(0.7183516073640783) q[3];
ry(0.003538872532457823) q[4];
rz(2.0468585065316613) q[4];
ry(-1.4800168447813424) q[5];
rz(-1.3662507381442772) q[5];
ry(-2.8338528695836724) q[6];
rz(1.7038214076716285) q[6];
ry(0.020886055186448822) q[7];
rz(2.4204901075564482) q[7];
ry(-0.027259455675171382) q[8];
rz(-3.1293367566497894) q[8];
ry(-3.116328686850615) q[9];
rz(-3.038067126589822) q[9];
ry(1.3563215433633171) q[10];
rz(1.4260338885351163) q[10];
ry(-0.7126013775802287) q[11];
rz(-0.769714467617459) q[11];
ry(1.5707227275051654) q[12];
rz(2.487512511967456) q[12];
ry(1.1906321211465292) q[13];
rz(0.9723352678193621) q[13];
ry(-3.1315064233723295) q[14];
rz(-2.9555331485457796) q[14];
ry(-3.1408557560686488) q[15];
rz(-1.8445457047857183) q[15];
ry(0.10080541911063376) q[16];
rz(-0.08971656405224893) q[16];
ry(1.2970478845999223) q[17];
rz(-0.005869611409489431) q[17];
ry(-1.8976825938725077) q[18];
rz(2.7986725805742463) q[18];
ry(0.10183761183741069) q[19];
rz(2.3325930545660047) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.5549882262072585) q[0];
rz(-1.26333235590623) q[0];
ry(-0.4028340413584722) q[1];
rz(-0.6259795423832837) q[1];
ry(0.6947797008218598) q[2];
rz(-2.7324160282871643) q[2];
ry(-3.091606202689014) q[3];
rz(-0.5549323671734578) q[3];
ry(-1.492998628786582) q[4];
rz(-1.6996529100289974) q[4];
ry(2.149743724478877) q[5];
rz(1.2337009824386291) q[5];
ry(-1.068845171377886) q[6];
rz(2.10770267967785) q[6];
ry(2.5372075592373893) q[7];
rz(2.7214981413334463) q[7];
ry(-2.598388401157104) q[8];
rz(0.23430388807650984) q[8];
ry(2.5724798299245175) q[9];
rz(-0.2196275717396956) q[9];
ry(2.993613144805644) q[10];
rz(1.4473811462209236) q[10];
ry(-2.8758626759716286) q[11];
rz(-2.493167312998496) q[11];
ry(-2.6951901255266213) q[12];
rz(-0.6503175663171393) q[12];
ry(-1.5708771858856707) q[13];
rz(1.9002391755969283) q[13];
ry(-1.3767622347463329) q[14];
rz(1.1517418401596164) q[14];
ry(-0.08248856553589601) q[15];
rz(-1.317776751664476) q[15];
ry(0.9366815871778175) q[16];
rz(2.091109064766821) q[16];
ry(-3.128075544352692) q[17];
rz(0.5964626228591037) q[17];
ry(-0.6251938874555719) q[18];
rz(2.2994366226200347) q[18];
ry(2.505874758951632) q[19];
rz(-0.2194371664879808) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(3.0551382491056884) q[0];
rz(-0.06607948252985807) q[0];
ry(-0.46289967785561004) q[1];
rz(-3.057261192762098) q[1];
ry(-2.55769065419643) q[2];
rz(-0.37255108038568974) q[2];
ry(0.05956673262355494) q[3];
rz(-2.4528730441114255) q[3];
ry(-2.712868175395048) q[4];
rz(-0.3683523245434621) q[4];
ry(-2.438257119734266) q[5];
rz(3.036672704321401) q[5];
ry(0.11072530696493002) q[6];
rz(3.0284353114974203) q[6];
ry(-2.7083566504828918) q[7];
rz(0.1828983126347534) q[7];
ry(-0.15030108678461165) q[8];
rz(-0.13746122146809547) q[8];
ry(0.03107656489537705) q[9];
rz(-0.596032534755147) q[9];
ry(1.8212885172914737) q[10];
rz(0.10696969023878465) q[10];
ry(3.113212944598438) q[11];
rz(-2.398428675321056) q[11];
ry(2.3759433498521494) q[12];
rz(2.3953721121043414) q[12];
ry(2.574559203524103) q[13];
rz(-0.6597827726150332) q[13];
ry(-1.761124575268517) q[14];
rz(0.20794723238114357) q[14];
ry(0.9166834334736622) q[15];
rz(2.499594033065707) q[15];
ry(0.028329824781086366) q[16];
rz(0.9518725894992577) q[16];
ry(3.134429863910828) q[17];
rz(-2.19888316959777) q[17];
ry(-2.8007123536419143) q[18];
rz(0.6766799098102096) q[18];
ry(2.3591729625552893) q[19];
rz(2.6232653054952415) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.007394879340852701) q[0];
rz(-0.3806545516456598) q[0];
ry(-1.174033510213751) q[1];
rz(-2.494020795641638) q[1];
ry(0.1669480350675201) q[2];
rz(1.240855766403851) q[2];
ry(3.126836642010214) q[3];
rz(-1.9106251150603253) q[3];
ry(-0.14745473974258963) q[4];
rz(-3.010854752499945) q[4];
ry(-3.1322023263671577) q[5];
rz(-0.37922006039796385) q[5];
ry(3.1322030090680246) q[6];
rz(1.3477585145700002) q[6];
ry(2.0320130740362314) q[7];
rz(2.5970377462279575) q[7];
ry(2.9961217368782465) q[8];
rz(0.6901975004566108) q[8];
ry(-3.0127372196968314) q[9];
rz(-0.15748574874918003) q[9];
ry(2.162327241715265) q[10];
rz(1.221049236408803) q[10];
ry(-2.1151347821198137) q[11];
rz(0.08148986315760735) q[11];
ry(-2.0492219406257135) q[12];
rz(2.0979691866796033) q[12];
ry(3.1408792882506713) q[13];
rz(2.3867027526204514) q[13];
ry(-0.0001884046570531781) q[14];
rz(2.0677838240262085) q[14];
ry(0.025859462587489383) q[15];
rz(1.2948534920172108) q[15];
ry(-2.8552463074316443) q[16];
rz(1.677037622755459) q[16];
ry(3.133272278690414) q[17];
rz(-2.2560795343560445) q[17];
ry(-0.9010017049384887) q[18];
rz(1.0430503486175993) q[18];
ry(2.7776881498950767) q[19];
rz(0.12823909071540385) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.3071537637800255) q[0];
rz(-0.791993081118612) q[0];
ry(0.8816123363410686) q[1];
rz(2.7009597188492385) q[1];
ry(-3.0783219379904554) q[2];
rz(-2.82114545982389) q[2];
ry(0.5088224817783454) q[3];
rz(1.7554517867906236) q[3];
ry(2.6591662965777423) q[4];
rz(0.926996092727389) q[4];
ry(0.5683969454984791) q[5];
rz(-1.0866182416201335) q[5];
ry(0.8705594204728007) q[6];
rz(-1.6060337579821033) q[6];
ry(-2.266889503614913) q[7];
rz(-0.8739271837224054) q[7];
ry(2.8185532652661527) q[8];
rz(-2.453186685799483) q[8];
ry(3.1347824862604656) q[9];
rz(-2.504199646061831) q[9];
ry(0.0007900664638107457) q[10];
rz(-1.035772114655253) q[10];
ry(-0.009858121882471593) q[11];
rz(0.07107707883726631) q[11];
ry(0.35359501378587427) q[12];
rz(-1.9727039515982208) q[12];
ry(1.4909661970187138) q[13];
rz(-0.48518832985835614) q[13];
ry(0.2882954733932691) q[14];
rz(-2.833296075831653) q[14];
ry(-1.0083370782595091) q[15];
rz(1.0811592379569002) q[15];
ry(1.6601259837598337) q[16];
rz(-2.703329210266497) q[16];
ry(-0.0003778531055704803) q[17];
rz(2.473021622373653) q[17];
ry(-1.5283956876025604) q[18];
rz(0.46681702700173133) q[18];
ry(0.13702221385648497) q[19];
rz(1.3885532949990202) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.5171110032511076) q[0];
rz(-1.7233681662764868) q[0];
ry(2.2014016719084255) q[1];
rz(1.0745766787102935) q[1];
ry(-0.012359673381843328) q[2];
rz(1.4736393750074646) q[2];
ry(-3.1298213774891877) q[3];
rz(-0.7691211186276969) q[3];
ry(-0.01995254836835336) q[4];
rz(2.7172932810920503) q[4];
ry(-0.039750452886869916) q[5];
rz(-2.711209734255504) q[5];
ry(-0.02324643599133875) q[6];
rz(2.2029165599700637) q[6];
ry(-2.276899203076699) q[7];
rz(2.292602890274979) q[7];
ry(-2.9878926107664) q[8];
rz(0.7467268952023156) q[8];
ry(-3.0224554189124797) q[9];
rz(2.3597003064167374) q[9];
ry(-2.373632833495493) q[10];
rz(3.0478318607480066) q[10];
ry(0.5236045020342935) q[11];
rz(-0.45370179706251784) q[11];
ry(-2.4977177090096463) q[12];
rz(1.0565064078890052) q[12];
ry(6.449123548701863e-05) q[13];
rz(-0.1935245707113502) q[13];
ry(-0.00047355996809938215) q[14];
rz(1.4405415747669956) q[14];
ry(1.5861548006941613) q[15];
rz(-2.8834636576528645) q[15];
ry(2.2524018608539995) q[16];
rz(-0.02249867017667865) q[16];
ry(-3.1412844206400097) q[17];
rz(0.6489797087111983) q[17];
ry(-1.2682028109761037) q[18];
rz(-2.566192081890246) q[18];
ry(0.6523341813607582) q[19];
rz(1.91554530872898) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.3674498915552236) q[0];
rz(1.993612799086507) q[0];
ry(-1.3924249512786933) q[1];
rz(-3.068905010151818) q[1];
ry(-0.8300576933742471) q[2];
rz(1.7558378944942827) q[2];
ry(1.1634812513558732) q[3];
rz(-1.324458935703686) q[3];
ry(0.9537429762688241) q[4];
rz(-1.0802391461559504) q[4];
ry(2.1832068406577774) q[5];
rz(2.7136497265415955) q[5];
ry(-2.3446614522219136) q[6];
rz(2.985115552947404) q[6];
ry(0.04423584865406169) q[7];
rz(-1.8188859356450564) q[7];
ry(2.4064584907023328) q[8];
rz(2.2280411434404366) q[8];
ry(-0.0933839619538599) q[9];
rz(1.1501565875029316) q[9];
ry(3.1381581243381897) q[10];
rz(-2.983921223369501) q[10];
ry(-1.1446819244856603) q[11];
rz(2.9459615473418768) q[11];
ry(-1.7651800940116251) q[12];
rz(0.3348135538920945) q[12];
ry(-0.6324397754547562) q[13];
rz(-0.8533512928274093) q[13];
ry(-0.7702513269287028) q[14];
rz(2.4768969193654176) q[14];
ry(-2.2248146425535724) q[15];
rz(-0.4216975087664814) q[15];
ry(1.6014182921119327) q[16];
rz(-1.0481490303609284) q[16];
ry(-0.001890364277137735) q[17];
rz(-1.420791996742112) q[17];
ry(-3.121493446630954) q[18];
rz(2.853061569629372) q[18];
ry(-2.479985414622667) q[19];
rz(-1.4463364162252814) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.418236746588457) q[0];
rz(-0.7142686910385998) q[0];
ry(0.26255811630463555) q[1];
rz(-3.021943215001218) q[1];
ry(-0.0009111505322388166) q[2];
rz(-2.5866792521769377) q[2];
ry(3.137457832389791) q[3];
rz(-1.601639664366898) q[3];
ry(3.070314739350285) q[4];
rz(-0.26251724769152573) q[4];
ry(-0.03262399314684927) q[5];
rz(-1.4781280747392156) q[5];
ry(-0.4071405480896695) q[6];
rz(-3.0680829657876907) q[6];
ry(0.02723475496330975) q[7];
rz(-0.5893063798172185) q[7];
ry(-3.1389364355268206) q[8];
rz(0.6842936351937174) q[8];
ry(0.9344607770767213) q[9];
rz(-0.8609678231737146) q[9];
ry(3.0741122231982896) q[10];
rz(0.9925316844214053) q[10];
ry(1.0391468715628802) q[11];
rz(-2.9833958236397913) q[11];
ry(0.07975605348964136) q[12];
rz(0.5006670303777467) q[12];
ry(-0.0033129517551287583) q[13];
rz(-0.6713105501116634) q[13];
ry(-3.1372256769377276) q[14];
rz(3.1385368781976077) q[14];
ry(2.2220650763788403) q[15];
rz(0.6731019385458186) q[15];
ry(-0.2783979294480011) q[16];
rz(-2.9882163821543686) q[16];
ry(0.025401162434811407) q[17];
rz(2.5284581816192846) q[17];
ry(0.4185780109691235) q[18];
rz(1.0447026452708217) q[18];
ry(-2.9585672227771194) q[19];
rz(-0.8099256853795376) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.3348453403699212) q[0];
rz(-0.04227216037965321) q[0];
ry(-3.1393146333029334) q[1];
rz(-0.793401429592774) q[1];
ry(-1.7298883824374487) q[2];
rz(-1.7699854056927908) q[2];
ry(-1.6282344553837182) q[3];
rz(-1.1999617313850406) q[3];
ry(-0.6587946295201119) q[4];
rz(-0.6951201748084577) q[4];
ry(-3.129243340915589) q[5];
rz(1.5484791779855729) q[5];
ry(1.0746292907772674) q[6];
rz(7.8045271147964e-05) q[6];
ry(3.141358097598586) q[7];
rz(-1.8833715986649981) q[7];
ry(1.6323228866598696) q[8];
rz(-2.5896238089205785) q[8];
ry(-2.358225504280293) q[9];
rz(-0.09051384285432221) q[9];
ry(-0.006257040708520312) q[10];
rz(1.5204156506576534) q[10];
ry(2.5318333041621344) q[11];
rz(-2.3789486218418383) q[11];
ry(1.456440328339541) q[12];
rz(-0.21834790856329264) q[12];
ry(-0.11457791402717472) q[13];
rz(-1.803651599963219) q[13];
ry(0.5778052064430419) q[14];
rz(-2.3884766501626307) q[14];
ry(-0.21296555512653637) q[15];
rz(-0.703012033575158) q[15];
ry(3.1321005637124184) q[16];
rz(-1.1844286474819021) q[16];
ry(-3.141365663669646) q[17];
rz(1.0459195607043341) q[17];
ry(-2.1959925397163316) q[18];
rz(-1.7555308038365531) q[18];
ry(2.5001574910065645) q[19];
rz(-0.6169252536746529) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-3.063244882091126) q[0];
rz(-0.7794102210948574) q[0];
ry(2.3932980421357963) q[1];
rz(1.1653435660393034) q[1];
ry(-0.0632928238729545) q[2];
rz(-2.3544887389649425) q[2];
ry(0.002636426971673133) q[3];
rz(-1.3730745699078122) q[3];
ry(-3.1333681533034614) q[4];
rz(1.344931796212248) q[4];
ry(-0.08439218534359719) q[5];
rz(-1.967859116843826) q[5];
ry(2.7471293300019917) q[6];
rz(1.9040414109126598) q[6];
ry(-0.005560381949080373) q[7];
rz(3.1011330864868976) q[7];
ry(-0.00022205201154568626) q[8];
rz(-0.29244814914831707) q[8];
ry(3.1162587427879873) q[9];
rz(1.9364944503241053) q[9];
ry(-0.007190941571465714) q[10];
rz(1.761931509114407) q[10];
ry(-0.19915598371143675) q[11];
rz(2.525924350301297) q[11];
ry(0.05728885060793482) q[12];
rz(2.8201204261614676) q[12];
ry(3.1340996403236847) q[13];
rz(2.952559735523907) q[13];
ry(0.006187347443999513) q[14];
rz(1.915073735114353) q[14];
ry(-0.6051805296998022) q[15];
rz(-1.2868512349147974) q[15];
ry(0.34058974802147723) q[16];
rz(2.4669490503628295) q[16];
ry(-0.5811199507274651) q[17];
rz(0.8666242252472931) q[17];
ry(-0.5462079191063625) q[18];
rz(-0.4606779835060273) q[18];
ry(1.8903436858279532) q[19];
rz(3.1111582071668993) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.033290934530700866) q[0];
rz(0.1794718471722989) q[0];
ry(0.841479649406863) q[1];
rz(2.075411101402115) q[1];
ry(-0.7682004969486087) q[2];
rz(-1.953175840044672) q[2];
ry(-0.34740695117456255) q[3];
rz(1.766889072070899) q[3];
ry(0.8459047446092227) q[4];
rz(2.5861483020675493) q[4];
ry(-0.918742155764062) q[5];
rz(0.6931626733638626) q[5];
ry(1.2096956358057516) q[6];
rz(-2.449780693212322) q[6];
ry(-0.18626643383178204) q[7];
rz(-2.8347996503063957) q[7];
ry(-1.6982030238265529) q[8];
rz(0.6698151835622124) q[8];
ry(0.29863639827485583) q[9];
rz(-1.7690433416048867) q[9];
ry(-3.138352996956671) q[10];
rz(-2.9629058399055026) q[10];
ry(2.0337407188546335) q[11];
rz(-3.021370035964964) q[11];
ry(0.8475658032432575) q[12];
rz(-1.6169689499102846) q[12];
ry(-2.8783554989500892) q[13];
rz(0.11186632208573238) q[13];
ry(3.035626934459921) q[14];
rz(0.30263777813811676) q[14];
ry(-2.4342163813672126) q[15];
rz(1.054515238988528) q[15];
ry(3.141502163804201) q[16];
rz(2.5600486930281874) q[16];
ry(-3.1402370703623554) q[17];
rz(-1.9523219926834985) q[17];
ry(-0.5552368278323883) q[18];
rz(-2.2270800664034844) q[18];
ry(0.7917233733033128) q[19];
rz(2.3264459650402296) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.2699495955996243) q[0];
rz(0.9396006995097173) q[0];
ry(-0.8145874099804421) q[1];
rz(0.822925915759793) q[1];
ry(0.1522031820490621) q[2];
rz(-2.1116482347562795) q[2];
ry(-3.1384180918659212) q[3];
rz(-1.2914128475374083) q[3];
ry(-3.140147997446169) q[4];
rz(-2.5344559299302967) q[4];
ry(-3.09545253321081) q[5];
rz(1.9857619817059433) q[5];
ry(0.010225747170044208) q[6];
rz(2.77662989635325) q[6];
ry(0.042244476935199955) q[7];
rz(-1.329837700015104) q[7];
ry(-3.115910786737335) q[8];
rz(3.109777025925939) q[8];
ry(-0.2766763226348558) q[9];
rz(1.1946372303543518) q[9];
ry(-3.041450341731999) q[10];
rz(1.8294635465326579) q[10];
ry(1.5125013052453797) q[11];
rz(0.16095662366259234) q[11];
ry(-0.43449603752053706) q[12];
rz(2.456312400598393) q[12];
ry(-0.0632970185594375) q[13];
rz(-1.7589076301148774) q[13];
ry(2.491496459414725) q[14];
rz(-3.0123375948550897) q[14];
ry(1.9609891180751386) q[15];
rz(-1.4878197935431934) q[15];
ry(-1.992332549039444) q[16];
rz(1.1698079946568285) q[16];
ry(-2.518048166530098) q[17];
rz(-0.4663808745475259) q[17];
ry(1.0922667837649946) q[18];
rz(-3.125960312751807) q[18];
ry(0.09962491792963495) q[19];
rz(-3.0454301521973814) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.909825455345174) q[0];
rz(1.05301801187349) q[0];
ry(0.4356646095256284) q[1];
rz(0.6511352736035719) q[1];
ry(-0.5643396049352569) q[2];
rz(1.6623616503038086) q[2];
ry(1.1797888618844043) q[3];
rz(2.291604425544805) q[3];
ry(-1.89654174288066) q[4];
rz(0.13137168299331484) q[4];
ry(-0.23456346682061113) q[5];
rz(0.9158849528920507) q[5];
ry(1.5478649391731167) q[6];
rz(0.805024163827096) q[6];
ry(-1.4999852604245891) q[7];
rz(0.05263748140252164) q[7];
ry(-0.13410151717348473) q[8];
rz(-1.6978851729977595) q[8];
ry(0.4098728550466566) q[9];
rz(-0.957019581680254) q[9];
ry(-0.00024270620041644261) q[10];
rz(-0.656396138038789) q[10];
ry(0.6327427576746338) q[11];
rz(1.8000211690153556) q[11];
ry(-2.328178526206417) q[12];
rz(0.12150545356143905) q[12];
ry(0.0927790623013106) q[13];
rz(-1.4377175927164219) q[13];
ry(-0.0879943042482969) q[14];
rz(-0.02109333425402063) q[14];
ry(-0.08915982689530193) q[15];
rz(-0.7392460212325955) q[15];
ry(-3.0615317223707255) q[16];
rz(-3.0107391978609117) q[16];
ry(-1.8946119193873863) q[17];
rz(3.1344936948871718) q[17];
ry(1.239866822332207) q[18];
rz(-2.83329000052385) q[18];
ry(1.1430199283813314) q[19];
rz(-1.6062292654271162) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.1607687393758376) q[0];
rz(-0.04324957556128428) q[0];
ry(0.1543850396300197) q[1];
rz(2.297406069659314) q[1];
ry(-1.5992435384492782) q[2];
rz(2.7124852585778663) q[2];
ry(-0.05339175283726213) q[3];
rz(1.5882502769249487) q[3];
ry(3.137047224691389) q[4];
rz(-1.1756770464161437) q[4];
ry(-0.04854566574535155) q[5];
rz(-1.4487121136267507) q[5];
ry(-0.010099250353672318) q[6];
rz(-0.4326107000452151) q[6];
ry(0.010225530301105401) q[7];
rz(-0.6290558241086908) q[7];
ry(-0.0012381439623661019) q[8];
rz(1.6581167944647468) q[8];
ry(1.0285779101184105) q[9];
rz(-1.5820547806643805) q[9];
ry(3.127428387747312) q[10];
rz(-1.1880984247300157) q[10];
ry(3.1020368545079076) q[11];
rz(-2.2258730611672557) q[11];
ry(0.005693339963973882) q[12];
rz(0.32876407405119057) q[12];
ry(0.014988898020002419) q[13];
rz(2.668999888477314) q[13];
ry(2.5205577091721523) q[14];
rz(2.0639906354738677) q[14];
ry(-3.019244097879299) q[15];
rz(2.584021780879711) q[15];
ry(3.0859998639016535) q[16];
rz(2.3469013639629095) q[16];
ry(-1.8357420292801074) q[17];
rz(-0.010345957720200083) q[17];
ry(-3.1193231302594384) q[18];
rz(-1.5723305271737464) q[18];
ry(-1.7476804756253168) q[19];
rz(-0.9023319678021018) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(1.0607514793793378) q[0];
rz(3.103451582704562) q[0];
ry(-1.5893059970192853) q[1];
rz(-3.0945443182648953) q[1];
ry(2.4976696100538516) q[2];
rz(-0.7383006315422076) q[2];
ry(0.04233861158989249) q[3];
rz(2.6985679269068212) q[3];
ry(-1.1977106405545266) q[4];
rz(2.429133616541813) q[4];
ry(2.9635442054711487) q[5];
rz(1.8853384761011462) q[5];
ry(2.6913160534417235) q[6];
rz(2.715256128159332) q[6];
ry(-2.9232370842127158) q[7];
rz(1.8514639080288369) q[7];
ry(0.37461065719372516) q[8];
rz(-2.3461327540221135) q[8];
ry(0.7449911992549695) q[9];
rz(2.7126767131207146) q[9];
ry(-2.866906419216033) q[10];
rz(-2.377009433486653) q[10];
ry(0.5667586042934094) q[11];
rz(-1.9648933890369307) q[11];
ry(-2.9171638939188607) q[12];
rz(-1.7510872845853873) q[12];
ry(1.9043830846598455) q[13];
rz(3.100730990256139) q[13];
ry(0.6714244662230096) q[14];
rz(2.5398617673044623) q[14];
ry(-1.9075634355178086) q[15];
rz(-1.5375934201036476) q[15];
ry(-0.003700129540689545) q[16];
rz(-1.088668269495588) q[16];
ry(2.7128631438838195) q[17];
rz(-3.1071424412043944) q[17];
ry(-1.572352530766267) q[18];
rz(2.833627340608973) q[18];
ry(1.6858111708616657) q[19];
rz(-1.82561252805802) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.5744088290800582) q[0];
rz(-0.039521938037768045) q[0];
ry(-3.0891428244626487) q[1];
rz(-2.9893551532617986) q[1];
ry(-0.02899680799586335) q[2];
rz(-2.2790710180583114) q[2];
ry(-3.0929138867386072) q[3];
rz(2.4487997960196637) q[3];
ry(3.136451289305975) q[4];
rz(1.1130242084786772) q[4];
ry(-0.002332064647581857) q[5];
rz(2.117902954800128) q[5];
ry(3.1404474603101007) q[6];
rz(2.2476712770040272) q[6];
ry(-0.01995100144699702) q[7];
rz(1.2258358217539465) q[7];
ry(0.0106549406358932) q[8];
rz(0.8497873839367535) q[8];
ry(-3.124935358634214) q[9];
rz(2.270076860245114) q[9];
ry(-0.009864825145730127) q[10];
rz(2.8136006366905537) q[10];
ry(0.1097144075626888) q[11];
rz(0.47395325261156074) q[11];
ry(3.1288968731886952) q[12];
rz(-0.031253317924571644) q[12];
ry(3.1375080689087023) q[13];
rz(-0.6526130067110331) q[13];
ry(0.000899808602015609) q[14];
rz(2.4399729781941777) q[14];
ry(3.134448713616968) q[15];
rz(0.07410228555642107) q[15];
ry(-3.0885151427432613) q[16];
rz(0.944519370390508) q[16];
ry(-1.4839589506061683) q[17];
rz(3.01129882105078) q[17];
ry(-3.0952358307026935) q[18];
rz(3.0581278431558556) q[18];
ry(1.5387874558748509) q[19];
rz(-2.937408104746484) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.8320198821512608) q[0];
rz(0.041862260899171784) q[0];
ry(2.948064568799305) q[1];
rz(0.1502287189360035) q[1];
ry(-1.2672064546561723) q[2];
rz(-0.21918532606347266) q[2];
ry(-2.3573454269885428) q[3];
rz(2.8936944333932546) q[3];
ry(2.4561419870846875) q[4];
rz(-2.957035299099654) q[4];
ry(0.3891904701376971) q[5];
rz(-2.924465600594464) q[5];
ry(1.876342498808515) q[6];
rz(-1.4146945475283224) q[6];
ry(-1.3596202321067843) q[7];
rz(0.2285484219165417) q[7];
ry(1.1736692482690723) q[8];
rz(3.053836834502188) q[8];
ry(1.3268460427906592) q[9];
rz(0.533281930096999) q[9];
ry(-0.5085806929381148) q[10];
rz(-2.251471846992783) q[10];
ry(0.3933509014084341) q[11];
rz(-3.0252016920468594) q[11];
ry(1.2101933236156368) q[12];
rz(0.46182637157742334) q[12];
ry(-1.968480463119497) q[13];
rz(-2.306066253726781) q[13];
ry(-2.259361075311108) q[14];
rz(-2.465461999325691) q[14];
ry(-1.5889099043524784) q[15];
rz(-2.739200429859591) q[15];
ry(-0.04280874031903566) q[16];
rz(2.247080967076881) q[16];
ry(0.0812752396204845) q[17];
rz(-2.910705903615164) q[17];
ry(-1.66030468088488) q[18];
rz(1.2561764284664019) q[18];
ry(1.840956649283755) q[19];
rz(-1.8581387714634552) q[19];