import argparse
import datetime

def obter_argumentos():
    """
    Analisa e retorna os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Processamento de eficiência hospitalar.")
    
    # Obter o mês anterior como padrão para 'competencia'
    hoje = datetime.date.today()
    primeiro_dia_mes_atual = hoje.replace(day=1)
    mes_anterior = primeiro_dia_mes_atual - datetime.timedelta(days=1)
    competencia_padrao = mes_anterior.strftime('%Y%m')

    parser.add_argument('--competencia', 
                        type=str, 
                        default=competencia_padrao,
                        help=f'Competência no formato AAAAMM (padrão: {competencia_padrao}).')
    
    parser.add_argument('--cnes', 
                        nargs='+', 
                        help='Lista de CNES para processar. Se não for fornecido, processa todos.')

    parser.add_argument('--n_jobs', 
                        type=int, 
                        default=-1, 
                        help='Número de processos paralelos (padrão: -1, usa todos os cores).')

    parser.add_argument('--forcar', 
                        action='store_true', 
                        help='Força o reprocessamento de CNES já existentes no cache.')

    return parser.parse_args() 