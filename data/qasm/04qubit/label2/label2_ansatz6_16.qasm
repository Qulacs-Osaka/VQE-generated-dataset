OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.4401393658834634) q[0];
ry(0.16798584929270893) q[1];
cx q[0],q[1];
ry(-0.9853212398410225) q[0];
ry(0.7497990622556504) q[1];
cx q[0],q[1];
ry(1.2534971999455475) q[1];
ry(-0.8765025991681039) q[2];
cx q[1],q[2];
ry(1.316658315104456) q[1];
ry(-1.7227643664481584) q[2];
cx q[1],q[2];
ry(-0.6586226675127957) q[2];
ry(-2.378104101750308) q[3];
cx q[2],q[3];
ry(-2.137465172563405) q[2];
ry(-1.4768999330397665) q[3];
cx q[2],q[3];
ry(-2.695012426094976) q[0];
ry(-1.039814990498842) q[1];
cx q[0],q[1];
ry(-1.9019931521716087) q[0];
ry(0.025297744509744678) q[1];
cx q[0],q[1];
ry(-1.861266202917175) q[1];
ry(0.46817522442732074) q[2];
cx q[1],q[2];
ry(-2.503624082645117) q[1];
ry(-0.6358883828731083) q[2];
cx q[1],q[2];
ry(0.32387441992802923) q[2];
ry(1.4545747214325255) q[3];
cx q[2],q[3];
ry(1.1444481149565646) q[2];
ry(-0.562615336245071) q[3];
cx q[2],q[3];
ry(2.55725666461777) q[0];
ry(-2.9542622611914866) q[1];
cx q[0],q[1];
ry(2.3972969188953925) q[0];
ry(-2.6338324617060622) q[1];
cx q[0],q[1];
ry(0.7832606284389388) q[1];
ry(-1.2207314838085388) q[2];
cx q[1],q[2];
ry(0.11590746520376172) q[1];
ry(2.0097407939867127) q[2];
cx q[1],q[2];
ry(1.8457037101594995) q[2];
ry(-1.2826707856564736) q[3];
cx q[2],q[3];
ry(-2.6080668806718976) q[2];
ry(0.028260967341165785) q[3];
cx q[2],q[3];
ry(2.1015112015217925) q[0];
ry(-0.1508905596792704) q[1];
cx q[0],q[1];
ry(-1.3878362964817024) q[0];
ry(2.6279454521855254) q[1];
cx q[0],q[1];
ry(-2.666108029762336) q[1];
ry(2.4720891293885323) q[2];
cx q[1],q[2];
ry(0.2520892687871532) q[1];
ry(-1.4598990906101141) q[2];
cx q[1],q[2];
ry(-0.4931014511209079) q[2];
ry(-1.5064398966420853) q[3];
cx q[2],q[3];
ry(2.029639345486628) q[2];
ry(-2.5798927930864504) q[3];
cx q[2],q[3];
ry(2.365775341904238) q[0];
ry(1.0240539453205078) q[1];
cx q[0],q[1];
ry(0.749532740778679) q[0];
ry(-0.4480308043159377) q[1];
cx q[0],q[1];
ry(1.3591935349331061) q[1];
ry(-1.6999276488085324) q[2];
cx q[1],q[2];
ry(1.5015153965793278) q[1];
ry(-0.39522340449658167) q[2];
cx q[1],q[2];
ry(-2.1526820739543586) q[2];
ry(-0.36950258176811007) q[3];
cx q[2],q[3];
ry(2.146889360898628) q[2];
ry(0.7937451206176545) q[3];
cx q[2],q[3];
ry(1.189195294287587) q[0];
ry(0.07986505819131007) q[1];
cx q[0],q[1];
ry(-1.9489734707097783) q[0];
ry(-2.725234470128417) q[1];
cx q[0],q[1];
ry(2.1603974555856715) q[1];
ry(2.6548190586258653) q[2];
cx q[1],q[2];
ry(0.7672740329618566) q[1];
ry(-1.6443106324664702) q[2];
cx q[1],q[2];
ry(2.304346202389828) q[2];
ry(-2.1948730252889406) q[3];
cx q[2],q[3];
ry(-1.3507582146990895) q[2];
ry(2.522914901843502) q[3];
cx q[2],q[3];
ry(-1.2306811079099704) q[0];
ry(3.1256345909186103) q[1];
cx q[0],q[1];
ry(0.5729456113956065) q[0];
ry(-0.30433111573220417) q[1];
cx q[0],q[1];
ry(0.2571376998971559) q[1];
ry(2.4198646249538704) q[2];
cx q[1],q[2];
ry(2.907745713515952) q[1];
ry(2.6216031108472575) q[2];
cx q[1],q[2];
ry(0.3233992821245423) q[2];
ry(-0.2662486812061715) q[3];
cx q[2],q[3];
ry(-1.4579223237951162) q[2];
ry(-1.1609742452226863) q[3];
cx q[2],q[3];
ry(0.45652025108985855) q[0];
ry(-1.5116867040221047) q[1];
cx q[0],q[1];
ry(-1.0506391704443676) q[0];
ry(2.345723439357988) q[1];
cx q[0],q[1];
ry(-1.3165110392681916) q[1];
ry(-1.0501393592528172) q[2];
cx q[1],q[2];
ry(-3.0967970257007207) q[1];
ry(2.5043285112024742) q[2];
cx q[1],q[2];
ry(0.32245406731377724) q[2];
ry(-1.7898842936590156) q[3];
cx q[2],q[3];
ry(-2.2656960046884587) q[2];
ry(1.0208463013975209) q[3];
cx q[2],q[3];
ry(-0.4149105944605332) q[0];
ry(2.6197757733555442) q[1];
cx q[0],q[1];
ry(0.554224306408929) q[0];
ry(2.0218745223209753) q[1];
cx q[0],q[1];
ry(-2.430831756271143) q[1];
ry(3.0458615117975114) q[2];
cx q[1],q[2];
ry(-3.034077677215391) q[1];
ry(1.1409631593960592) q[2];
cx q[1],q[2];
ry(-2.0462594209148435) q[2];
ry(1.2661804656728624) q[3];
cx q[2],q[3];
ry(1.0368801259873441) q[2];
ry(3.1070042399157236) q[3];
cx q[2],q[3];
ry(-1.2797293133242178) q[0];
ry(-1.9893230537465478) q[1];
cx q[0],q[1];
ry(2.602047598717707) q[0];
ry(2.2226754811830665) q[1];
cx q[0],q[1];
ry(-3.133781700936686) q[1];
ry(2.973578115055909) q[2];
cx q[1],q[2];
ry(-1.2801510117947636) q[1];
ry(-2.7383202071493122) q[2];
cx q[1],q[2];
ry(-1.2934747920771894) q[2];
ry(-2.8092765128553467) q[3];
cx q[2],q[3];
ry(1.902630924402172) q[2];
ry(3.1166420693255783) q[3];
cx q[2],q[3];
ry(-1.5881797939476445) q[0];
ry(0.1379281865941274) q[1];
cx q[0],q[1];
ry(-1.2570047975034662) q[0];
ry(-1.801508948190216) q[1];
cx q[0],q[1];
ry(-2.403326136248401) q[1];
ry(2.021301144484704) q[2];
cx q[1],q[2];
ry(-0.135391942912397) q[1];
ry(-2.6099072270692387) q[2];
cx q[1],q[2];
ry(-2.730765490792397) q[2];
ry(-2.838136449581609) q[3];
cx q[2],q[3];
ry(0.05944560611626219) q[2];
ry(-1.7547201332644278) q[3];
cx q[2],q[3];
ry(-2.968783425368165) q[0];
ry(-1.5112013899900563) q[1];
cx q[0],q[1];
ry(0.915494042454231) q[0];
ry(2.283311023820258) q[1];
cx q[0],q[1];
ry(1.3529185893576667) q[1];
ry(2.4420600951201252) q[2];
cx q[1],q[2];
ry(-0.2750066951139045) q[1];
ry(-0.6572890829010909) q[2];
cx q[1],q[2];
ry(1.572082098626871) q[2];
ry(-0.7847850924351052) q[3];
cx q[2],q[3];
ry(-1.6554374596187849) q[2];
ry(-0.1526179339136844) q[3];
cx q[2],q[3];
ry(-2.453441548058356) q[0];
ry(1.1844385926457734) q[1];
cx q[0],q[1];
ry(-1.4746122248746736) q[0];
ry(2.6589149013059608) q[1];
cx q[0],q[1];
ry(2.446664089685737) q[1];
ry(-2.722872060830921) q[2];
cx q[1],q[2];
ry(0.17882660880964166) q[1];
ry(2.1978950594973257) q[2];
cx q[1],q[2];
ry(-2.3741118622300785) q[2];
ry(-0.37550180172965975) q[3];
cx q[2],q[3];
ry(-1.9263190660494098) q[2];
ry(-2.5558232535127265) q[3];
cx q[2],q[3];
ry(-0.44772442706450466) q[0];
ry(2.0034215142019045) q[1];
cx q[0],q[1];
ry(1.4959260410274557) q[0];
ry(-1.6940511016380935) q[1];
cx q[0],q[1];
ry(0.7892524439013545) q[1];
ry(-1.6223237988378498) q[2];
cx q[1],q[2];
ry(2.7303392215830615) q[1];
ry(-3.0432557162948837) q[2];
cx q[1],q[2];
ry(0.8207392393090488) q[2];
ry(2.3553641395916474) q[3];
cx q[2],q[3];
ry(0.2218052236833188) q[2];
ry(1.6791288875146355) q[3];
cx q[2],q[3];
ry(1.7248932363115577) q[0];
ry(-0.9816799951567369) q[1];
cx q[0],q[1];
ry(-1.7013067978205325) q[0];
ry(-0.9271637162132942) q[1];
cx q[0],q[1];
ry(-1.0381549967092487) q[1];
ry(-0.7136955599514896) q[2];
cx q[1],q[2];
ry(-0.22389021811433268) q[1];
ry(3.056343602388718) q[2];
cx q[1],q[2];
ry(-1.3122462157484138) q[2];
ry(0.4140337659623299) q[3];
cx q[2],q[3];
ry(3.0579431230754226) q[2];
ry(-2.8096239558248675) q[3];
cx q[2],q[3];
ry(2.1922957133256253) q[0];
ry(-0.704863660931041) q[1];
cx q[0],q[1];
ry(2.010041584842014) q[0];
ry(2.424969490780003) q[1];
cx q[0],q[1];
ry(3.1141651187798325) q[1];
ry(1.6020110624542945) q[2];
cx q[1],q[2];
ry(-1.0089095684730538) q[1];
ry(-2.2545752096761524) q[2];
cx q[1],q[2];
ry(2.129112583393136) q[2];
ry(-1.6918496735271447) q[3];
cx q[2],q[3];
ry(-2.7338854100707453) q[2];
ry(-1.9003882771690566) q[3];
cx q[2],q[3];
ry(-0.6251455610796954) q[0];
ry(0.6591062235382448) q[1];
cx q[0],q[1];
ry(0.22338483670493733) q[0];
ry(0.605575184248768) q[1];
cx q[0],q[1];
ry(2.864270245736426) q[1];
ry(-1.1060755165985219) q[2];
cx q[1],q[2];
ry(-1.945407338537943) q[1];
ry(1.6530413690383026) q[2];
cx q[1],q[2];
ry(-2.7890014152011884) q[2];
ry(-2.617986439360946) q[3];
cx q[2],q[3];
ry(-0.8524227037704931) q[2];
ry(3.0737424906735673) q[3];
cx q[2],q[3];
ry(2.6313512771527647) q[0];
ry(1.5651630341034384) q[1];
cx q[0],q[1];
ry(2.3864136934718134) q[0];
ry(0.07690589473560072) q[1];
cx q[0],q[1];
ry(1.1753933802905316) q[1];
ry(1.302415208499044) q[2];
cx q[1],q[2];
ry(2.0334727466714906) q[1];
ry(2.0668703393284993) q[2];
cx q[1],q[2];
ry(0.1644805872509868) q[2];
ry(2.7565088594303755) q[3];
cx q[2],q[3];
ry(1.9117704016858799) q[2];
ry(-2.753051168891566) q[3];
cx q[2],q[3];
ry(-0.7378370448214402) q[0];
ry(-2.295891897812416) q[1];
cx q[0],q[1];
ry(2.453231605763319) q[0];
ry(-1.1382342226928541) q[1];
cx q[0],q[1];
ry(0.46620442900010445) q[1];
ry(1.5381544486427408) q[2];
cx q[1],q[2];
ry(0.9864738617340132) q[1];
ry(-0.8750028913481388) q[2];
cx q[1],q[2];
ry(-0.17826029277810607) q[2];
ry(-0.08586005709680312) q[3];
cx q[2],q[3];
ry(-0.7456768051794382) q[2];
ry(-3.0559957421163655) q[3];
cx q[2],q[3];
ry(-1.787227630586516) q[0];
ry(3.044395488602981) q[1];
ry(-1.8649227714105665) q[2];
ry(-2.163942424247802) q[3];