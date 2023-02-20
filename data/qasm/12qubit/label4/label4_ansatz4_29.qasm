OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.4970654020989054) q[0];
rz(1.505417650444124) q[0];
ry(-0.9225420271646875) q[1];
rz(0.65393130609593) q[1];
ry(0.0009428255829355692) q[2];
rz(0.41650638438321935) q[2];
ry(3.139841666080415) q[3];
rz(-1.41221406410659) q[3];
ry(-1.5749687391897478) q[4];
rz(-0.05694096798824333) q[4];
ry(-1.5681950919427723) q[5];
rz(-0.8721100450112037) q[5];
ry(-0.0988696274597544) q[6];
rz(-0.5860021869287877) q[6];
ry(-0.1783712112181446) q[7];
rz(2.5915926706198764) q[7];
ry(-2.4887016075985624) q[8];
rz(-2.867415624498093) q[8];
ry(-1.3197566232046365) q[9];
rz(2.3567802762575396) q[9];
ry(0.34010403983401094) q[10];
rz(1.0101717528456695) q[10];
ry(-0.9925094482759567) q[11];
rz(1.971192044872566) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.6832353004394554) q[0];
rz(1.8589706736024052) q[0];
ry(-0.24131491700413168) q[1];
rz(-1.9297366525624293) q[1];
ry(-1.3966092347481265) q[2];
rz(-2.9873156163336065) q[2];
ry(-1.5088679365599478) q[3];
rz(-3.052365722247929) q[3];
ry(1.1914487995558547) q[4];
rz(2.608603857543118) q[4];
ry(3.0616629267320414) q[5];
rz(0.3463169458005355) q[5];
ry(-1.1737724907363338) q[6];
rz(-2.985928811288177) q[6];
ry(-1.354380299324814) q[7];
rz(2.313198634335442) q[7];
ry(-3.0360566003811487) q[8];
rz(1.2543454792028663) q[8];
ry(-0.7636302388175036) q[9];
rz(2.452656689892347) q[9];
ry(2.249065107499738) q[10];
rz(0.06862399579270925) q[10];
ry(2.0706522303256336) q[11];
rz(0.181848427002957) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.041576249019407) q[0];
rz(-0.9113015427795288) q[0];
ry(1.6170507129185197) q[1];
rz(3.1206144189760403) q[1];
ry(3.074939020540003) q[2];
rz(-1.494836381505861) q[2];
ry(3.0659075214876603) q[3];
rz(2.7308345122819246) q[3];
ry(3.1413064952995975) q[4];
rz(-0.7897245166489947) q[4];
ry(-3.141189382102402) q[5];
rz(-1.6563082187096523) q[5];
ry(-1.6624009160203526) q[6];
rz(2.957752072322571) q[6];
ry(-1.7370102701541608) q[7];
rz(3.0975908689131124) q[7];
ry(3.1307223797464485) q[8];
rz(-1.2483509304536256) q[8];
ry(1.4628692675336632) q[9];
rz(-2.9004556681413347) q[9];
ry(-1.1616062474063056) q[10];
rz(2.6178874250295663) q[10];
ry(0.19576009294726404) q[11];
rz(-0.9239613306900207) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.0678559899665079) q[0];
rz(-0.9643518057108009) q[0];
ry(-2.172641239005465) q[1];
rz(-2.3522558496527872) q[1];
ry(-0.8717278022698737) q[2];
rz(-1.14383362234974) q[2];
ry(-0.04385580144616696) q[3];
rz(-2.722146320548177) q[3];
ry(1.4117650898225707) q[4];
rz(0.5328681572940885) q[4];
ry(-1.6854859510174318) q[5];
rz(2.8011559813501203) q[5];
ry(-1.092383600160387) q[6];
rz(1.90992261214602) q[6];
ry(1.7883392161431226) q[7];
rz(-2.47188321852111) q[7];
ry(-2.3864956172906693) q[8];
rz(2.3742364884786964) q[8];
ry(-2.2467657374974914) q[9];
rz(0.5983050770025365) q[9];
ry(-1.410856357505872) q[10];
rz(-0.605020757233401) q[10];
ry(0.9135215624290436) q[11];
rz(-0.41559320368379815) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.4084620050659233) q[0];
rz(0.9172493184196808) q[0];
ry(-1.7944142433629109) q[1];
rz(2.6608056853831283) q[1];
ry(-0.6024140783300458) q[2];
rz(-1.8780622023178157) q[2];
ry(-2.524887910221373) q[3];
rz(1.5218422062597163) q[3];
ry(3.1283202943391366) q[4];
rz(-1.1262441322824692) q[4];
ry(-3.1262046260520675) q[5];
rz(-2.058578354404419) q[5];
ry(-1.3645145332341133) q[6];
rz(1.5910562144764406) q[6];
ry(0.6749895282550336) q[7];
rz(3.006408251831947) q[7];
ry(1.6699174250175526) q[8];
rz(1.678884862373918) q[8];
ry(-0.9379111845877275) q[9];
rz(0.34004870350609484) q[9];
ry(2.2420408760080672) q[10];
rz(0.21089111913714498) q[10];
ry(-2.0404669343025317) q[11];
rz(2.7989207479994405) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2486344822024158) q[0];
rz(0.7395277286690196) q[0];
ry(1.6623141307096747) q[1];
rz(2.984710389488887) q[1];
ry(-1.2195685039818187) q[2];
rz(-2.6328462486615303) q[2];
ry(-1.4917182858574218) q[3];
rz(2.615033642684985) q[3];
ry(3.096248204664496) q[4];
rz(-2.7946462271172483) q[4];
ry(-3.0918675418797292) q[5];
rz(-0.09034056232796851) q[5];
ry(0.3285569916636856) q[6];
rz(1.1903986089824885) q[6];
ry(-2.5027605235018466) q[7];
rz(0.9003836700671395) q[7];
ry(-0.33745741276685237) q[8];
rz(-2.7234596040561514) q[8];
ry(-1.4547116381007905) q[9];
rz(2.0880383424770286) q[9];
ry(0.8655017514324205) q[10];
rz(-1.4120905455916202) q[10];
ry(0.6701102346789698) q[11];
rz(0.8631402099707133) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.0652760750095616) q[0];
rz(-0.4794405706597206) q[0];
ry(1.9280048255695785) q[1];
rz(-0.6153613994012108) q[1];
ry(0.26587647566949923) q[2];
rz(-1.1566989339445337) q[2];
ry(2.861531977961301) q[3];
rz(2.2288772776590555) q[3];
ry(0.0126975841604402) q[4];
rz(3.0387812787132593) q[4];
ry(-0.012998469911386949) q[5];
rz(2.9321584275867414) q[5];
ry(-1.7725450992349645) q[6];
rz(-1.1365778599430811) q[6];
ry(0.7795664384272447) q[7];
rz(3.0045728401319636) q[7];
ry(0.3213527983026161) q[8];
rz(1.9076709375075849) q[8];
ry(-0.7292788727549295) q[9];
rz(2.2956943649302004) q[9];
ry(-3.0625028869704516) q[10];
rz(-0.8089090428793335) q[10];
ry(-0.5562962084507068) q[11];
rz(1.430220030185163) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2988063080451098) q[0];
rz(-2.451338512067594) q[0];
ry(-2.577929840199517) q[1];
rz(-1.7919269681540086) q[1];
ry(-2.563378125530798) q[2];
rz(-0.17826862875818783) q[2];
ry(2.237952285457448) q[3];
rz(-0.6827468389309761) q[3];
ry(-0.03695640478982796) q[4];
rz(3.0693488982711403) q[4];
ry(-0.020525380059219025) q[5];
rz(-2.87551398801642) q[5];
ry(-0.9498870825520402) q[6];
rz(1.033170277127476) q[6];
ry(1.1924731698021844) q[7];
rz(-2.5914133788192544) q[7];
ry(-0.8666358175199392) q[8];
rz(2.352602958440248) q[8];
ry(-1.6208075599173712) q[9];
rz(2.513189339805099) q[9];
ry(-0.8499239464165943) q[10];
rz(-2.8273574253814195) q[10];
ry(1.500516942452105) q[11];
rz(-0.8731641750901651) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.1924166584918823) q[0];
rz(2.8378034542631685) q[0];
ry(0.688697361355172) q[1];
rz(-0.21637776910145948) q[1];
ry(-1.521760089534765) q[2];
rz(-1.564400833520592) q[2];
ry(1.4545939937072587) q[3];
rz(1.3963720825980506) q[3];
ry(3.1236791433435265) q[4];
rz(2.294845365941293) q[4];
ry(-3.0334963992409176) q[5];
rz(1.7595503188580446) q[5];
ry(1.1311118428078164) q[6];
rz(1.4867176534712325) q[6];
ry(-0.789262132757445) q[7];
rz(2.285712939077802) q[7];
ry(1.1287501798473862) q[8];
rz(-0.46117150731470685) q[8];
ry(1.4142645978469304) q[9];
rz(0.8231608050620238) q[9];
ry(1.3499393934846402) q[10];
rz(-1.2584865046317948) q[10];
ry(-1.2416710343802226) q[11];
rz(0.5100666341229543) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.7389258455736876) q[0];
rz(0.5426612219990633) q[0];
ry(-1.4111144092689933) q[1];
rz(-2.86599676354267) q[1];
ry(1.854133717834179) q[2];
rz(-2.940056666854591) q[2];
ry(-1.1829545147367373) q[3];
rz(-3.0386192604334132) q[3];
ry(0.007341873476258698) q[4];
rz(1.2587217316975838) q[4];
ry(-0.01209354929261064) q[5];
rz(-1.9049073719014054) q[5];
ry(-2.926902449216184) q[6];
rz(0.8456203300530568) q[6];
ry(1.0851165034350272) q[7];
rz(0.08278658067427448) q[7];
ry(-0.7176849436852866) q[8];
rz(-0.47781519814238754) q[8];
ry(2.117039947416286) q[9];
rz(3.1367244366353044) q[9];
ry(0.0023439981850625472) q[10];
rz(2.771517181335212) q[10];
ry(-0.7983267694122268) q[11];
rz(2.834279825680636) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.420836970313999) q[0];
rz(0.14398910085255281) q[0];
ry(-1.1563726789830726) q[1];
rz(-1.0697148993577472) q[1];
ry(-1.2668589128567442) q[2];
rz(-1.8635658129436026) q[2];
ry(1.8412003749485937) q[3];
rz(1.1557007602931222) q[3];
ry(-0.16631836214216758) q[4];
rz(2.15986637686475) q[4];
ry(-0.15741920275036939) q[5];
rz(-2.2700983896239975) q[5];
ry(0.1358262473701754) q[6];
rz(-2.148069008758876) q[6];
ry(-1.320184392106011) q[7];
rz(3.121890620802821) q[7];
ry(-1.86880789905699) q[8];
rz(-1.0368047793242796) q[8];
ry(-1.1421445233022949) q[9];
rz(1.758181358976817) q[9];
ry(-0.6313329614717406) q[10];
rz(0.0919728250148763) q[10];
ry(1.357508493564514) q[11];
rz(1.8315882238189831) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.2775468867942834) q[0];
rz(1.832721400127588) q[0];
ry(1.3681240968807868) q[1];
rz(1.337250401987462) q[1];
ry(2.3838933055751004) q[2];
rz(-0.22652949559027835) q[2];
ry(-2.2531325937082576) q[3];
rz(3.020000381860885) q[3];
ry(-0.046983513082180295) q[4];
rz(2.541655240508083) q[4];
ry(0.04526620711233331) q[5];
rz(-0.5391055283424242) q[5];
ry(-2.1258761395949852) q[6];
rz(2.2890985372511516) q[6];
ry(1.0875419628063734) q[7];
rz(1.4962983561526269) q[7];
ry(1.3614383508460421) q[8];
rz(1.6109704724602256) q[8];
ry(1.6468252818228803) q[9];
rz(-2.389997408280389) q[9];
ry(-0.44087945356205926) q[10];
rz(2.619101015044967) q[10];
ry(-1.6889677588990653) q[11];
rz(-0.38195368148029196) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.1032626161564494) q[0];
rz(-2.0201323829771427) q[0];
ry(-1.5691681181256634) q[1];
rz(2.2753818560293766) q[1];
ry(0.8252127046404638) q[2];
rz(0.4219705416542789) q[2];
ry(-0.7788645178168148) q[3];
rz(2.185951137461691) q[3];
ry(0.07572668634818225) q[4];
rz(0.2696846535055606) q[4];
ry(1.6495009805376268) q[5];
rz(1.2626744730435124) q[5];
ry(-2.1890706138745717) q[6];
rz(-2.603725104149046) q[6];
ry(0.6168917845768016) q[7];
rz(0.41969969259951295) q[7];
ry(-1.6752120397587538) q[8];
rz(-2.0664265480521236) q[8];
ry(1.7255951542230568) q[9];
rz(-1.3547074069006353) q[9];
ry(-2.2314843270669376) q[10];
rz(1.8451311272983324) q[10];
ry(0.8510493599005448) q[11];
rz(-0.8785759511528963) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.9417465160784135) q[0];
rz(-1.9652038400208962) q[0];
ry(-1.73574895248918) q[1];
rz(2.507969524345476) q[1];
ry(-2.8444307680573466) q[2];
rz(2.412853278063622) q[2];
ry(2.8872915286654215) q[3];
rz(-0.27798912002341003) q[3];
ry(3.1117461351515274) q[4];
rz(1.1488987632750016) q[4];
ry(3.1142308383187927) q[5];
rz(0.3712422066601002) q[5];
ry(-0.006850340128473454) q[6];
rz(-1.843190212417615) q[6];
ry(0.002635140400546021) q[7];
rz(1.0606571186066533) q[7];
ry(-2.7192236817013815) q[8];
rz(-2.432926989048655) q[8];
ry(-0.8719243450251652) q[9];
rz(0.392936896504513) q[9];
ry(-2.6303332108407607) q[10];
rz(1.0479709184382078) q[10];
ry(1.8682443819692) q[11];
rz(2.9436219175849687) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.6026168991516379) q[0];
rz(1.228984518943911) q[0];
ry(-1.5733400199164196) q[1];
rz(-2.8364791081018894) q[1];
ry(2.8881097702565213) q[2];
rz(1.3521585761708712) q[2];
ry(0.23096026413447035) q[3];
rz(-2.7684401621314847) q[3];
ry(-1.6388874239698772) q[4];
rz(-2.0476490259140205) q[4];
ry(-0.15668897635072643) q[5];
rz(0.5180731181785235) q[5];
ry(-0.3065504782024382) q[6];
rz(2.2614232301682895) q[6];
ry(1.0102975302679482) q[7];
rz(-2.0955508037846786) q[7];
ry(0.062394835613549526) q[8];
rz(-3.0618687536756943) q[8];
ry(0.18520358999821873) q[9];
rz(0.8491861508332229) q[9];
ry(2.2985488642034175) q[10];
rz(2.1087842250953264) q[10];
ry(0.8680822286959167) q[11];
rz(1.1740043355392165) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.0049819221803062) q[0];
rz(2.384071708679301) q[0];
ry(-1.4052172616547016) q[1];
rz(-0.14708976488241188) q[1];
ry(0.025241640325858405) q[2];
rz(-1.1703735299857385) q[2];
ry(0.0028126804165626846) q[3];
rz(-3.1100913291044474) q[3];
ry(3.122177730139891) q[4];
rz(1.8080831177611678) q[4];
ry(3.1277309948393444) q[5];
rz(0.04913531043426822) q[5];
ry(1.6015696453958306) q[6];
rz(-1.4183717898270913) q[6];
ry(2.043347912078394) q[7];
rz(-1.2228732619709852) q[7];
ry(-1.0729378020665212) q[8];
rz(0.17366266429559563) q[8];
ry(1.702362240431642) q[9];
rz(0.6616672973972851) q[9];
ry(-1.5503246779060698) q[10];
rz(-0.9275490330740928) q[10];
ry(0.6518719811639251) q[11];
rz(-0.07237796810750387) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.2818673561899274) q[0];
rz(-2.409725015478531) q[0];
ry(-3.055680512591724) q[1];
rz(-2.1611291472032446) q[1];
ry(-0.636647988460707) q[2];
rz(1.4814930834641675) q[2];
ry(0.804037740395687) q[3];
rz(-1.4145328204673842) q[3];
ry(-3.127577149529743) q[4];
rz(0.4791899546902) q[4];
ry(0.015466743919381989) q[5];
rz(-2.96356865593036) q[5];
ry(3.0362049300355003) q[6];
rz(-1.3776500472325566) q[6];
ry(0.13968821840587692) q[7];
rz(-1.9658592074230645) q[7];
ry(-0.3118595531244657) q[8];
rz(1.219700525510496) q[8];
ry(3.0898064422604725) q[9];
rz(-2.8701495004461206) q[9];
ry(2.89552768441214) q[10];
rz(-0.47311275055984325) q[10];
ry(0.9335912547564966) q[11];
rz(-0.8550864802906935) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.7458274924293828) q[0];
rz(-0.028920360923066178) q[0];
ry(-1.556531746083284) q[1];
rz(-1.578904786565989) q[1];
ry(-2.995085781369873) q[2];
rz(2.2419732025700885) q[2];
ry(-3.040272679732784) q[3];
rz(-1.858794049191272) q[3];
ry(-0.056295468597615056) q[4];
rz(-0.32858346132148686) q[4];
ry(0.03954066399614413) q[5];
rz(2.193413087049114) q[5];
ry(-1.4601854447411684) q[6];
rz(0.5897233078240901) q[6];
ry(1.0607833850313853) q[7];
rz(2.6300334198136364) q[7];
ry(-1.4949673054242378) q[8];
rz(-0.13888284778764604) q[8];
ry(-0.4600268840457869) q[9];
rz(-2.5199650347178375) q[9];
ry(-0.6430345245599605) q[10];
rz(-1.9412579410250999) q[10];
ry(-0.602462251049177) q[11];
rz(-0.9930341313482012) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.0183269540406625) q[0];
rz(-0.45281414952135535) q[0];
ry(-0.8743822051965331) q[1];
rz(2.314599626633885) q[1];
ry(2.661212779811391) q[2];
rz(0.13080615042194943) q[2];
ry(0.17580858495717008) q[3];
rz(0.2401708373425127) q[3];
ry(-1.2206601554690024) q[4];
rz(1.4132715779471043) q[4];
ry(2.5318067773775947) q[5];
rz(-1.6618698502255143) q[5];
ry(-2.618289196608268) q[6];
rz(1.0937422099448542) q[6];
ry(-2.9437527004411597) q[7];
rz(-1.6625408315853751) q[7];
ry(1.3798583682321397) q[8];
rz(-2.244481814988438) q[8];
ry(1.7516571691921539) q[9];
rz(1.346262871383364) q[9];
ry(-1.1974822148676392) q[10];
rz(0.10873412562121178) q[10];
ry(1.3361057508923182) q[11];
rz(2.274343854325865) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8835473303127546) q[0];
rz(2.6794521962120665) q[0];
ry(2.22367439389734) q[1];
rz(-1.0068875992644142) q[1];
ry(0.9207538587565898) q[2];
rz(2.692283823884847) q[2];
ry(0.8067628394158319) q[3];
rz(-2.713876312085492) q[3];
ry(-0.0026563986563731172) q[4];
rz(-0.371662423154893) q[4];
ry(0.022859393478038705) q[5];
rz(-0.3245086271979822) q[5];
ry(-0.007414398123127163) q[6];
rz(0.8069528409577672) q[6];
ry(-0.020053988112162215) q[7];
rz(-2.8794494215054276) q[7];
ry(-2.284476317641824) q[8];
rz(1.019310929452116) q[8];
ry(2.8623675251344736) q[9];
rz(0.8494876209645685) q[9];
ry(-1.4373004473904771) q[10];
rz(1.482503691714454) q[10];
ry(1.53106719080738) q[11];
rz(-1.2484447811837676) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.7662952614237533) q[0];
rz(-2.415452647889079) q[0];
ry(-1.576046188013822) q[1];
rz(-3.0793055727725798) q[1];
ry(2.4020719938675823) q[2];
rz(-0.028912548002735315) q[2];
ry(-1.3252768472944005) q[3];
rz(1.1786988245733472) q[3];
ry(0.9148288404532847) q[4];
rz(0.3088406939857613) q[4];
ry(1.59478817907506) q[5];
rz(1.3920260707514567) q[5];
ry(-1.3754307715229726) q[6];
rz(1.3665841374252135) q[6];
ry(0.34331986949944665) q[7];
rz(0.9471196798106973) q[7];
ry(-2.9821925068245174) q[8];
rz(-3.0216723971746045) q[8];
ry(-0.35864543942903027) q[9];
rz(-2.9813940130342518) q[9];
ry(-0.6587030408250701) q[10];
rz(1.8039228960987024) q[10];
ry(0.7540241347222469) q[11];
rz(-2.2273412766581053) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.004602013042991793) q[0];
rz(1.6104017136243798) q[0];
ry(3.0674362486368763) q[1];
rz(0.0008099126588652082) q[1];
ry(3.125593936681527) q[2];
rz(-1.4375404139069694) q[2];
ry(-0.004172198843138775) q[3];
rz(1.6808197828746045) q[3];
ry(0.0006320502310153131) q[4];
rz(-2.554843842980388) q[4];
ry(4.338446041264718e-05) q[5];
rz(-2.930582350019799) q[5];
ry(-0.08062580760200344) q[6];
rz(-0.6803626265713012) q[6];
ry(0.1574272928476219) q[7];
rz(1.8505461936925256) q[7];
ry(3.0425973394662678) q[8];
rz(0.801163376796635) q[8];
ry(-1.5332468034228266) q[9];
rz(-0.793288982735998) q[9];
ry(2.1917798076491044) q[10];
rz(2.3085668631014786) q[10];
ry(-2.7189051117004825) q[11];
rz(1.4874228568488403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.0595992679491397) q[0];
rz(-3.070402335083771) q[0];
ry(-1.6329098762991068) q[1];
rz(-0.18251278117771189) q[1];
ry(-0.47726251416940446) q[2];
rz(-1.8636700444785113) q[2];
ry(-2.039950761683407) q[3];
rz(-0.41322129639021643) q[3];
ry(-2.9967929672947156) q[4];
rz(-2.0934736575833495) q[4];
ry(0.025542879418774866) q[5];
rz(-2.1499837217506714) q[5];
ry(1.6074355122535255) q[6];
rz(1.1634225872686619) q[6];
ry(-2.0248031318787847) q[7];
rz(2.0577922504434847) q[7];
ry(-2.6068860472228437) q[8];
rz(2.3912346460927734) q[8];
ry(-0.25000627754832383) q[9];
rz(1.351851018204805) q[9];
ry(2.1885704145442144) q[10];
rz(-1.4846522479715878) q[10];
ry(-0.779029635395777) q[11];
rz(2.631471300706064) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.241589367670704) q[0];
rz(1.6290022466076923) q[0];
ry(1.1379357339613834) q[1];
rz(-1.2357164869766544) q[1];
ry(1.775548886517611) q[2];
rz(-2.0916655542318034) q[2];
ry(-2.7214573778033855) q[3];
rz(-1.7825125008253657) q[3];
ry(-1.574399802965714) q[4];
rz(-1.7060768817970793) q[4];
ry(-0.009134962136866598) q[5];
rz(-2.3947274780581917) q[5];
ry(1.5724432306888811) q[6];
rz(1.4189026390939727) q[6];
ry(1.6367157355845645) q[7];
rz(1.6162903363195573) q[7];
ry(1.5727175821173836) q[8];
rz(0.14987675082203247) q[8];
ry(-1.669260877796738) q[9];
rz(0.44873186336653165) q[9];
ry(1.93556045614679) q[10];
rz(0.06984917212763517) q[10];
ry(2.250820366880755) q[11];
rz(3.0774771495759503) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.012713258671182004) q[0];
rz(-0.4462712961096589) q[0];
ry(0.2317689546687133) q[1];
rz(2.9101944551482277) q[1];
ry(3.1401562324570107) q[2];
rz(1.134419286022819) q[2];
ry(-7.895646351041563e-05) q[3];
rz(-1.3696897898482225) q[3];
ry(0.001613331329878953) q[4];
rz(-0.27623205721055205) q[4];
ry(3.138052969927051) q[5];
rz(1.562052237194561) q[5];
ry(-1.562661164016295) q[6];
rz(-0.01493586751898578) q[6];
ry(1.573757513927246) q[7];
rz(0.007887366232639848) q[7];
ry(1.4380684347524193) q[8];
rz(-3.102610718590494) q[8];
ry(-0.05438736405739164) q[9];
rz(2.0521746461231993) q[9];
ry(-1.837227209292983) q[10];
rz(0.574261246362182) q[10];
ry(-0.9520486040095448) q[11];
rz(0.9454284353867103) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8645159602434438) q[0];
rz(-0.206594661556641) q[0];
ry(1.5562155343466466) q[1];
rz(-0.4473935890778349) q[1];
ry(0.9075429478795564) q[2];
rz(-2.903495297739418) q[2];
ry(-2.6851860847118143) q[3];
rz(-2.5155927610982824) q[3];
ry(-0.006697316780397422) q[4];
rz(1.7725954342740362) q[4];
ry(0.00208546093602191) q[5];
rz(0.8160248448486629) q[5];
ry(1.5685671087828332) q[6];
rz(3.139939360458953) q[6];
ry(1.569734937848927) q[7];
rz(-1.2978510663711218) q[7];
ry(-1.3337449708311022) q[8];
rz(1.609649860580439) q[8];
ry(2.4402628594613094) q[9];
rz(1.112817955131967) q[9];
ry(0.24020287055333675) q[10];
rz(-0.20801786618931128) q[10];
ry(1.1404882987220029) q[11];
rz(0.9109893244037338) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.584737249239092) q[0];
rz(0.093816856888313) q[0];
ry(1.8823983238141837) q[1];
rz(-1.550464689763284) q[1];
ry(-3.1403727572475724) q[2];
rz(-1.1918139247080886) q[2];
ry(-0.007474362914726072) q[3];
rz(1.6248975698632808) q[3];
ry(3.1407645904470747) q[4];
rz(2.3522043689119907) q[4];
ry(1.5451837120232945) q[5];
rz(0.268197394162776) q[5];
ry(-2.7367294231922767) q[6];
rz(0.008757042963179806) q[6];
ry(-2.302297540268362) q[7];
rz(-1.2020599633540527) q[7];
ry(-1.7721918890197899) q[8];
rz(0.3409048624466821) q[8];
ry(1.6004410968767668) q[9];
rz(-3.0893832743612335) q[9];
ry(0.9176694985597225) q[10];
rz(-2.2210407446914973) q[10];
ry(-0.5029185055889052) q[11];
rz(1.0078837169599648) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(2.706446460137566) q[0];
rz(-3.139650975763066) q[0];
ry(-2.6781069684863836) q[1];
rz(-1.5924334986486102) q[1];
ry(-1.3000852595003576) q[2];
rz(2.4118130202486836) q[2];
ry(1.5504667100272664) q[3];
rz(1.6177053392287069) q[3];
ry(-0.006610231733538396) q[4];
rz(3.1073652142438384) q[4];
ry(3.1295123904258024) q[5];
rz(1.6044836528353421) q[5];
ry(1.5683951576321693) q[6];
rz(-1.5680619734076098) q[6];
ry(-1.5747881699717483) q[7];
rz(2.399858567396858) q[7];
ry(2.8029973638146095) q[8];
rz(-1.1855191583258122) q[8];
ry(0.033668729195494074) q[9];
rz(-1.361159521044149) q[9];
ry(0.7280800995427912) q[10];
rz(2.5045824608176677) q[10];
ry(-1.0332925444561836) q[11];
rz(0.4882035278273422) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.7418464234269553) q[0];
rz(0.042768906687267114) q[0];
ry(0.035400704282412904) q[1];
rz(1.1596028654020947) q[1];
ry(-3.1407913329997914) q[2];
rz(-2.784558031399752) q[2];
ry(-1.5729739034351162) q[3];
rz(-0.9663315263454864) q[3];
ry(1.5792762767864854) q[4];
rz(-3.1372553725987293) q[4];
ry(-3.1303332747914467) q[5];
rz(-1.8275048745707914) q[5];
ry(-2.420293267123082) q[6];
rz(0.665776864114563) q[6];
ry(-3.1386671551702743) q[7];
rz(2.4357582642080273) q[7];
ry(1.0771609208225956) q[8];
rz(-2.427450096689956) q[8];
ry(3.0520232311263813) q[9];
rz(-2.5132650519916027) q[9];
ry(1.4017194205507193) q[10];
rz(-1.0443839626635576) q[10];
ry(2.1940516519052284) q[11];
rz(-0.33773055106806216) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.572653886193165) q[0];
rz(-0.004195998900715203) q[0];
ry(-0.0005377170983607015) q[1];
rz(0.6537315080562616) q[1];
ry(0.00041760047485596955) q[2];
rz(2.0616344889897773) q[2];
ry(3.1406965566438036) q[3];
rz(2.2555877871961565) q[3];
ry(1.5701559226188868) q[4];
rz(-3.141387553821048) q[4];
ry(1.5713702572292056) q[5];
rz(0.0038977012056014004) q[5];
ry(0.004539970183799369) q[6];
rz(-2.2298525554186224) q[6];
ry(-0.006032240374712394) q[7];
rz(-0.2003308642993858) q[7];
ry(1.6146308118093273) q[8];
rz(0.6031346374000277) q[8];
ry(1.426735311251327) q[9];
rz(0.3089346513732784) q[9];
ry(-2.0566975350219305) q[10];
rz(2.33412779572125) q[10];
ry(1.7351590632734795) q[11];
rz(1.6276340776044647) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.3448975153016527) q[0];
rz(1.5732891991539322) q[0];
ry(1.5771258340126606) q[1];
rz(3.1400172402258577) q[1];
ry(1.5709109798276228) q[2];
rz(-1.569965982632735) q[2];
ry(-0.0009450344261260692) q[3];
rz(3.060313232721456) q[3];
ry(1.5704785121202423) q[4];
rz(-2.1621857887795155) q[4];
ry(1.5683122955931257) q[5];
rz(-2.581506533560053) q[5];
ry(-0.5115933399401156) q[6];
rz(-1.5736229649308673) q[6];
ry(3.129128686149402) q[7];
rz(1.405587951095053) q[7];
ry(2.6499204847625877) q[8];
rz(2.7143192858169205) q[8];
ry(3.041305480747224) q[9];
rz(2.412154164336677) q[9];
ry(-1.834924181740565) q[10];
rz(-0.31948263149209577) q[10];
ry(1.9196889272142827) q[11];
rz(-3.0818408155733237) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.569385718465509) q[0];
rz(-1.5708518915733036) q[0];
ry(-1.5724059672322763) q[1];
rz(-1.5711157022612592) q[1];
ry(-1.5695860486384356) q[2];
rz(3.1404963071473144) q[2];
ry(-1.5703796662132012) q[3];
rz(-2.6760701791645314) q[3];
ry(3.1394690958888325) q[4];
rz(1.07486171733044) q[4];
ry(-3.0978370217826257) q[5];
rz(-0.05376136632808265) q[5];
ry(-1.5690497688341152) q[6];
rz(3.1016039830329243) q[6];
ry(1.5733745429634451) q[7];
rz(1.5684803976331656) q[7];
ry(-0.03495789756180656) q[8];
rz(-1.7032361535845024) q[8];
ry(1.8671414315868808) q[9];
rz(-3.011092739655273) q[9];
ry(0.2635642966561642) q[10];
rz(1.4229826283481186) q[10];
ry(-1.7376537385216526) q[11];
rz(-0.994150695759065) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5678731217817512) q[0];
rz(0.3186418123376883) q[0];
ry(-1.569913015578663) q[1];
rz(-1.0331795119176201) q[1];
ry(-1.5710033434179558) q[2];
rz(-2.824710103254381) q[2];
ry(-3.1385776726180743) q[3];
rz(-2.140264761650369) q[3];
ry(-0.005704679536220636) q[4];
rz(0.2329910076189048) q[4];
ry(-1.5647567653729146) q[5];
rz(-2.607199817434203) q[5];
ry(-1.559911792930417) q[6];
rz(0.2879051176592363) q[6];
ry(-1.694747557858099) q[7];
rz(-2.6104660228527448) q[7];
ry(0.03445852321374687) q[8];
rz(1.7209372765488649) q[8];
ry(-3.0517419555953578) q[9];
rz(-2.542967711012505) q[9];
ry(1.546520040672255) q[10];
rz(-2.797471575787468) q[10];
ry(1.6482255760019613) q[11];
rz(0.5061117229777009) q[11];