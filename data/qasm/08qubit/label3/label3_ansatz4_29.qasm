OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.1814046102166225) q[0];
rz(-2.563228341796167) q[0];
ry(1.4824152430919513) q[1];
rz(-1.66650369316028) q[1];
ry(-0.6012625551747026) q[2];
rz(1.0331540028897823) q[2];
ry(-0.9704701059332912) q[3];
rz(-0.45263010052115293) q[3];
ry(1.592363682077183) q[4];
rz(-2.9039299839411283) q[4];
ry(-1.3752867119729046) q[5];
rz(-0.7658954546360854) q[5];
ry(1.6424894683997557) q[6];
rz(1.2475486893035288) q[6];
ry(1.7025495079888007) q[7];
rz(-1.901747703453479) q[7];
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
ry(-0.551574727881237) q[0];
rz(-2.2310106973459227) q[0];
ry(1.0771782255439466) q[1];
rz(-1.257783423118904) q[1];
ry(-0.7350975000434373) q[2];
rz(0.47675269138737164) q[2];
ry(-1.4892841106796102) q[3];
rz(-1.6515701876332562) q[3];
ry(-0.4979551213422233) q[4];
rz(1.7257162753672193) q[4];
ry(-1.0311213545060118) q[5];
rz(-2.763547956869947) q[5];
ry(-2.821063499693612) q[6];
rz(-0.930980381995289) q[6];
ry(-0.9207637214187043) q[7];
rz(-1.3105966635700907) q[7];
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
ry(2.0228749747735253) q[0];
rz(-2.9918505491217675) q[0];
ry(-1.5945536618085168) q[1];
rz(-1.2200123495258604) q[1];
ry(-1.9640866265880232) q[2];
rz(-1.609895539395538) q[2];
ry(0.17611586168029358) q[3];
rz(-3.079170498432702) q[3];
ry(-0.589095141004374) q[4];
rz(-0.4006134309734394) q[4];
ry(2.99924002601805) q[5];
rz(-1.506565058314245) q[5];
ry(2.612923619819018) q[6];
rz(-2.774239683353665) q[6];
ry(-0.7414725530073394) q[7];
rz(-2.807091007877493) q[7];
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
ry(0.2972002834227322) q[0];
rz(-0.11881216631039243) q[0];
ry(2.237843231509054) q[1];
rz(1.546626416786225) q[1];
ry(-0.046737308916869225) q[2];
rz(-1.9428875037284388) q[2];
ry(-2.495154523288266) q[3];
rz(0.3097433245530922) q[3];
ry(-1.5852343789644205) q[4];
rz(-0.9477592347280726) q[4];
ry(2.4439275360978145) q[5];
rz(2.8396014796208036) q[5];
ry(1.5610304369754449) q[6];
rz(-2.7007014351723972) q[6];
ry(-2.7913365211271537) q[7];
rz(-2.7478544906976525) q[7];
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
ry(1.516558320006241) q[0];
rz(1.7140503689094062) q[0];
ry(1.8313504892197408) q[1];
rz(-0.7891634210834221) q[1];
ry(2.3373088837775366) q[2];
rz(1.4920303360495364) q[2];
ry(-0.838477992774056) q[3];
rz(-2.984563706524878) q[3];
ry(2.595880482021266) q[4];
rz(1.3897370458912806) q[4];
ry(1.526737123113561) q[5];
rz(-2.3186746836908894) q[5];
ry(3.1059632229044754) q[6];
rz(0.4670097950814469) q[6];
ry(1.4055102599976335) q[7];
rz(1.0047281665434615) q[7];
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
ry(-1.2163865551160624) q[0];
rz(0.5964829465808429) q[0];
ry(-2.9212571827138345) q[1];
rz(0.650789824555445) q[1];
ry(-0.9001148549081353) q[2];
rz(1.8434566808006732) q[2];
ry(2.567362862495666) q[3];
rz(-0.7141774572837614) q[3];
ry(1.3371091976162814) q[4];
rz(-1.1832601519738122) q[4];
ry(-2.7423627845277463) q[5];
rz(-0.07771751914468794) q[5];
ry(-2.777431935971337) q[6];
rz(-2.778154251338244) q[6];
ry(-1.8802691893966639) q[7];
rz(-0.8583837185081897) q[7];
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
ry(1.4279608545507663) q[0];
rz(-2.7420321505806946) q[0];
ry(-1.4456363091339484) q[1];
rz(3.0343583117527992) q[1];
ry(0.05041652029255537) q[2];
rz(-2.718753113249745) q[2];
ry(-2.8320088986120187) q[3];
rz(2.1778396681757006) q[3];
ry(-1.9253422887992997) q[4];
rz(2.674001261762364) q[4];
ry(2.2000284508237122) q[5];
rz(2.6507513228734543) q[5];
ry(1.6201251590837427) q[6];
rz(1.7049613266017722) q[6];
ry(-1.879548818808513) q[7];
rz(2.593050768012448) q[7];
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
ry(1.5301762447013365) q[0];
rz(2.0773597542176883) q[0];
ry(-0.44674554499567937) q[1];
rz(2.7953477549078314) q[1];
ry(2.5097830416119735) q[2];
rz(2.267635039277076) q[2];
ry(-2.421950078197021) q[3];
rz(-1.4965251879071158) q[3];
ry(0.21608492699644027) q[4];
rz(-2.191395776414538) q[4];
ry(2.999697587067274) q[5];
rz(2.7336282050190226) q[5];
ry(-1.316803769675138) q[6];
rz(-1.467795130854974) q[6];
ry(-0.645846741824034) q[7];
rz(-1.3742453523251292) q[7];
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
ry(-2.9090740226518297) q[0];
rz(-1.6750943510568648) q[0];
ry(0.9531774951751296) q[1];
rz(-0.2620625940254677) q[1];
ry(0.8721917110252599) q[2];
rz(3.070455662277496) q[2];
ry(-2.5724586222612955) q[3];
rz(-0.3620717958261386) q[3];
ry(-2.3907834482466765) q[4];
rz(0.08299622630189654) q[4];
ry(1.182353250147556) q[5];
rz(-2.90284185454465) q[5];
ry(-1.5997288403775327) q[6];
rz(-1.710045508782303) q[6];
ry(0.0687328654870063) q[7];
rz(0.7762043410930648) q[7];
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
ry(2.6731076331008086) q[0];
rz(-3.068766065179618) q[0];
ry(-1.00320891283688) q[1];
rz(-1.3642339226405322) q[1];
ry(-2.5787860380754193) q[2];
rz(0.9448903377866451) q[2];
ry(1.558067711706965) q[3];
rz(-3.0202778342004555) q[3];
ry(2.929561618085954) q[4];
rz(-0.807647715047965) q[4];
ry(1.7268743019485298) q[5];
rz(-0.11970736955682781) q[5];
ry(-0.668025698122463) q[6];
rz(1.868076796331827) q[6];
ry(0.04889812891784565) q[7];
rz(1.5510793204405495) q[7];
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
ry(0.8557844962252714) q[0];
rz(-2.6428522083562673) q[0];
ry(-1.8817413054609877) q[1];
rz(-1.5893566884681893) q[1];
ry(0.09002876107207404) q[2];
rz(-2.1788149386263864) q[2];
ry(0.5427806496712111) q[3];
rz(-2.3628813526008106) q[3];
ry(-2.0257187198898388) q[4];
rz(-0.7842152840556144) q[4];
ry(-1.4196019150246242) q[5];
rz(0.7515049031094652) q[5];
ry(-1.6054961011150257) q[6];
rz(1.143236086998824) q[6];
ry(-1.330957270515932) q[7];
rz(2.212272183932936) q[7];
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
ry(0.4948724487563841) q[0];
rz(-1.6879640250772088) q[0];
ry(-2.5986288103334703) q[1];
rz(0.5197849102881715) q[1];
ry(2.7329921522935807) q[2];
rz(-1.5566052388869656) q[2];
ry(0.7007648545304042) q[3];
rz(-3.0328264843104993) q[3];
ry(-1.619404020232427) q[4];
rz(-2.854631866733454) q[4];
ry(-1.1089211960732202) q[5];
rz(1.3440643645727137) q[5];
ry(-2.938239435006777) q[6];
rz(-1.882829512933621) q[6];
ry(-2.619204194997467) q[7];
rz(2.70187997990071) q[7];
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
ry(1.6727641821285006) q[0];
rz(-1.1470127938016526) q[0];
ry(-0.4060823328301888) q[1];
rz(-2.4666407834362962) q[1];
ry(-0.46309124791124123) q[2];
rz(-0.37181855091095617) q[2];
ry(-0.24967576850908743) q[3];
rz(-1.360809952884056) q[3];
ry(-0.36100395361331566) q[4];
rz(-2.1769531280444405) q[4];
ry(-0.031951356981690715) q[5];
rz(0.6691292282518003) q[5];
ry(2.8999147724671372) q[6];
rz(0.44581536448208237) q[6];
ry(2.3820177642381473) q[7];
rz(0.23612080432572213) q[7];
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
ry(-1.989581814148037) q[0];
rz(-0.5573668157852422) q[0];
ry(1.1572104044026588) q[1];
rz(1.3819624004914264) q[1];
ry(1.8107630192543611) q[2];
rz(-3.039455751301832) q[2];
ry(-0.23354393115299565) q[3];
rz(1.1141326182568037) q[3];
ry(-2.2435225373765584) q[4];
rz(0.015624430054614004) q[4];
ry(2.0897984828981895) q[5];
rz(1.3663871444155715) q[5];
ry(-1.453881333059405) q[6];
rz(1.7878593385563928) q[6];
ry(1.4171063584815409) q[7];
rz(1.5333897299513026) q[7];
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
ry(0.9029492936681658) q[0];
rz(-1.269960611570071) q[0];
ry(-2.670849094397736) q[1];
rz(-2.828224285501014) q[1];
ry(-2.3785014279814125) q[2];
rz(1.6384707021657174) q[2];
ry(1.2594868708534983) q[3];
rz(0.7932318385481841) q[3];
ry(-0.8366710967397512) q[4];
rz(0.768779637337927) q[4];
ry(0.6948377671427295) q[5];
rz(-3.062168892977951) q[5];
ry(-3.0549221495746566) q[6];
rz(1.623661085024497) q[6];
ry(-1.690210577506243) q[7];
rz(-2.9654964660652867) q[7];
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
ry(0.9462971100487029) q[0];
rz(-2.538843076597791) q[0];
ry(0.8909173464892799) q[1];
rz(-0.9200555570300335) q[1];
ry(-1.9986873099144669) q[2];
rz(2.1594328425123854) q[2];
ry(-2.928505875112007) q[3];
rz(2.1415254315563566) q[3];
ry(-2.4780554834801003) q[4];
rz(-1.9902066398388571) q[4];
ry(-0.862261145750904) q[5];
rz(-1.6225493777936437) q[5];
ry(-1.8459638637995543) q[6];
rz(-1.596905659735912) q[6];
ry(-2.3084706056564217) q[7];
rz(1.3960162459356014) q[7];
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
ry(-2.004502834397205) q[0];
rz(-0.5948273055914175) q[0];
ry(-0.18937944576119037) q[1];
rz(-0.5109244109337274) q[1];
ry(-1.7528671580868993) q[2];
rz(-2.0283601986356765) q[2];
ry(1.6812202699602186) q[3];
rz(1.6010097387328024) q[3];
ry(-1.7033180799726249) q[4];
rz(-2.581870005720584) q[4];
ry(2.1842644944856247) q[5];
rz(-1.6416030920582463) q[5];
ry(-2.1567248683207456) q[6];
rz(-1.6212259701248923) q[6];
ry(-1.1381055591888867) q[7];
rz(2.416895490972231) q[7];
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
ry(0.11398410189844982) q[0];
rz(1.2569999486319479) q[0];
ry(0.5369865964573722) q[1];
rz(-2.591662240218428) q[1];
ry(-2.442452278109966) q[2];
rz(-2.855659583426598) q[2];
ry(-1.8839483791748215) q[3];
rz(1.4360233910824027) q[3];
ry(-2.222088880680344) q[4];
rz(-2.9590349460133707) q[4];
ry(-1.7956745519203112) q[5];
rz(2.4453924876484154) q[5];
ry(-2.4206811763676446) q[6];
rz(-1.3442663794708762) q[6];
ry(2.086172414360016) q[7];
rz(-1.0098093450882584) q[7];
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
ry(1.13235735534614) q[0];
rz(-0.8044402509378168) q[0];
ry(-0.592839217628624) q[1];
rz(1.2147846097551396) q[1];
ry(-0.355910099406548) q[2];
rz(2.855789475199835) q[2];
ry(0.36270556934741993) q[3];
rz(-2.5996568900453645) q[3];
ry(-2.042425095238981) q[4];
rz(-2.155633992854133) q[4];
ry(2.519869634295447) q[5];
rz(-1.4892464425340792) q[5];
ry(-1.0431605154033217) q[6];
rz(-1.9951758238496906) q[6];
ry(0.6472969253531166) q[7];
rz(-0.5176003871502664) q[7];
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
ry(2.8300813445020885) q[0];
rz(-1.7384229121381694) q[0];
ry(2.2710299519387984) q[1];
rz(-2.9887315910622925) q[1];
ry(0.5297890482009264) q[2];
rz(-1.8249557649456207) q[2];
ry(1.9135502748854074) q[3];
rz(2.216253073478692) q[3];
ry(3.120943772593286) q[4];
rz(-1.0938988404649663) q[4];
ry(-0.6829847337683389) q[5];
rz(2.825776294605962) q[5];
ry(1.1875738885990537) q[6];
rz(-0.02252345833240188) q[6];
ry(-2.4361647248569147) q[7];
rz(0.9282737177441395) q[7];
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
ry(-3.0449779269038855) q[0];
rz(2.0106288716527874) q[0];
ry(-1.6121335357311324) q[1];
rz(-2.2092452032961267) q[1];
ry(2.8652120193283728) q[2];
rz(-2.8591488756847983) q[2];
ry(0.8673442638792954) q[3];
rz(-0.30184138575696906) q[3];
ry(1.409910118516475) q[4];
rz(-2.0802070871541085) q[4];
ry(-1.8619979079544642) q[5];
rz(-1.837333044570431) q[5];
ry(-0.2048575994584523) q[6];
rz(2.2128452494905773) q[6];
ry(0.8329115424928486) q[7];
rz(-2.6720468180346013) q[7];
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
ry(-1.3218259638941294) q[0];
rz(1.3076986565226887) q[0];
ry(1.536872289864704) q[1];
rz(2.1164580050978286) q[1];
ry(0.7890438768811148) q[2];
rz(-2.508431373829003) q[2];
ry(-2.921115162391505) q[3];
rz(-0.34058730303593876) q[3];
ry(0.2814409648199394) q[4];
rz(2.7934491675338147) q[4];
ry(1.3048532675310114) q[5];
rz(-2.3140946381052703) q[5];
ry(2.3694484609842186) q[6];
rz(2.69761149132254) q[6];
ry(-0.1329732950280027) q[7];
rz(-3.0513415544715734) q[7];
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
ry(0.17614961138364896) q[0];
rz(-2.3934800433520182) q[0];
ry(-2.824672577882174) q[1];
rz(-1.2176770063535125) q[1];
ry(-1.6505919596861367) q[2];
rz(0.3349059785606209) q[2];
ry(0.36008072762241383) q[3];
rz(-0.2996873585712605) q[3];
ry(-1.9915409269419708) q[4];
rz(-2.611699948648216) q[4];
ry(-1.9384961657705775) q[5];
rz(0.716714398854176) q[5];
ry(-1.0683489984602508) q[6];
rz(-0.5691242613030205) q[6];
ry(-1.9530982409917135) q[7];
rz(2.6498887207377804) q[7];
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
ry(0.5146701912508188) q[0];
rz(1.7973466582563629) q[0];
ry(-0.12224370962117888) q[1];
rz(-1.61232309164249) q[1];
ry(-0.7652561166755625) q[2];
rz(-2.3835726909574206) q[2];
ry(-1.236148415430037) q[3];
rz(1.069116189914542) q[3];
ry(2.9221231526232048) q[4];
rz(-1.8251033447294758) q[4];
ry(-1.9560769004683418) q[5];
rz(-2.3381670996950654) q[5];
ry(-1.7361530953783852) q[6];
rz(0.5379850764721947) q[6];
ry(-1.9892363730459612) q[7];
rz(-0.5233092071578787) q[7];
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
ry(0.08826301088467581) q[0];
rz(-0.444591502559962) q[0];
ry(-1.8322787825195492) q[1];
rz(1.739662120210106) q[1];
ry(-1.7160563812578615) q[2];
rz(2.844021228708914) q[2];
ry(-2.7398398598039906) q[3];
rz(2.009735061053733) q[3];
ry(-2.355040980713889) q[4];
rz(0.7549161815233978) q[4];
ry(-0.5622570091412133) q[5];
rz(0.8698340998143756) q[5];
ry(0.7155192135374042) q[6];
rz(-1.7106719706148912) q[6];
ry(0.3412763782024859) q[7];
rz(-0.2311105865468453) q[7];
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
ry(-1.1557260255833104) q[0];
rz(2.5579760423864126) q[0];
ry(0.5390835312711149) q[1];
rz(-2.190539635310061) q[1];
ry(-1.6867394733761794) q[2];
rz(-1.5977429175406757) q[2];
ry(-1.2671638627133504) q[3];
rz(2.2193433040443207) q[3];
ry(1.9173370227379811) q[4];
rz(0.18185003585292212) q[4];
ry(1.1314390179810265) q[5];
rz(-2.764238596275337) q[5];
ry(1.8534812774078526) q[6];
rz(-2.7637234553075993) q[6];
ry(-0.08134718954539238) q[7];
rz(1.9380347886787113) q[7];
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
ry(2.130628550299342) q[0];
rz(2.70057246596434) q[0];
ry(0.9564983336730273) q[1];
rz(2.6984144487474047) q[1];
ry(1.0590177097327205) q[2];
rz(-1.5690241688125204) q[2];
ry(-2.8204302676804605) q[3];
rz(1.6877911859815722) q[3];
ry(-2.1815485132758967) q[4];
rz(-2.454158524208891) q[4];
ry(-1.2718192774605352) q[5];
rz(2.64933357306671) q[5];
ry(-2.5933020469187293) q[6];
rz(0.039251494872981844) q[6];
ry(-2.020127731388368) q[7];
rz(0.4348039488172022) q[7];
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
ry(0.5035528644046031) q[0];
rz(0.796122001236455) q[0];
ry(1.0313543497518438) q[1];
rz(-2.517633513271458) q[1];
ry(1.6109521957543134) q[2];
rz(-0.39018139577075234) q[2];
ry(2.1867418897175375) q[3];
rz(-1.808677087822726) q[3];
ry(-2.282049934023857) q[4];
rz(-1.2769048692492104) q[4];
ry(-0.9389611493700785) q[5];
rz(0.4063344153714654) q[5];
ry(-0.4940523702338732) q[6];
rz(0.3490641300235966) q[6];
ry(3.081809517739554) q[7];
rz(-0.989483538584241) q[7];
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
ry(-0.4374415463579024) q[0];
rz(-0.8963348486890927) q[0];
ry(-0.5954462140427284) q[1];
rz(0.28363471233382676) q[1];
ry(-1.1488797203397798) q[2];
rz(-2.7334443549060605) q[2];
ry(0.7293194532821464) q[3];
rz(0.07053517870762427) q[3];
ry(-2.223534810832529) q[4];
rz(-0.5525742648118639) q[4];
ry(-0.37156025249261165) q[5];
rz(-3.034455238642933) q[5];
ry(-2.514805237394145) q[6];
rz(-2.2770610386531165) q[6];
ry(2.5601422803329554) q[7];
rz(1.6753330602480627) q[7];
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
ry(1.0460078597370632) q[0];
rz(1.6083019598473587) q[0];
ry(-0.4053453406697774) q[1];
rz(-0.39792753578323214) q[1];
ry(2.046897668520377) q[2];
rz(-1.7892280767822113) q[2];
ry(1.7166136847471225) q[3];
rz(-0.8722665876259063) q[3];
ry(1.7036985799015254) q[4];
rz(-1.8768626276316172) q[4];
ry(-0.22073712583502805) q[5];
rz(-0.8354577609732782) q[5];
ry(0.09270728400745955) q[6];
rz(2.2239118516412946) q[6];
ry(-1.0622635628090755) q[7];
rz(-1.6502302182226112) q[7];
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
ry(2.411971256172164) q[0];
rz(-2.043227677431079) q[0];
ry(-1.0203599227760685) q[1];
rz(1.4616923611836858) q[1];
ry(-2.9195739344533425) q[2];
rz(-0.11693169813978925) q[2];
ry(-1.0927803006162808) q[3];
rz(0.9915616256112746) q[3];
ry(2.5567373451564754) q[4];
rz(3.0227953939418875) q[4];
ry(-1.3018636107368158) q[5];
rz(1.7788928317146566) q[5];
ry(-0.977461246337488) q[6];
rz(0.9298551589989784) q[6];
ry(2.463354956853202) q[7];
rz(-0.3139857845824205) q[7];
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
ry(0.8310580846497109) q[0];
rz(2.0394284952778863) q[0];
ry(1.1112737081656103) q[1];
rz(2.9927127644243074) q[1];
ry(-2.383043860062335) q[2];
rz(1.2799439112037445) q[2];
ry(-0.18925529885547204) q[3];
rz(2.791076151752946) q[3];
ry(-2.426029060692792) q[4];
rz(-0.7740123987375662) q[4];
ry(2.4828213330541216) q[5];
rz(-2.3129483400141715) q[5];
ry(-1.104671361576301) q[6];
rz(-2.421228876655102) q[6];
ry(-2.904600342911033) q[7];
rz(1.2971790811905528) q[7];
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
ry(-0.5270219132628884) q[0];
rz(-2.9301284612685707) q[0];
ry(-0.4984173729352177) q[1];
rz(-2.4492997833079366) q[1];
ry(0.1575629538377088) q[2];
rz(2.2802638607978185) q[2];
ry(2.8820759720530313) q[3];
rz(-0.11482713928297913) q[3];
ry(-0.7569736340468181) q[4];
rz(-1.1771436681161775) q[4];
ry(3.052871457673556) q[5];
rz(0.5224073149656041) q[5];
ry(3.0862440786331398) q[6];
rz(0.6910110497573647) q[6];
ry(2.9447764508531464) q[7];
rz(1.3735372549769507) q[7];