import ast
import copy
import json
import os.path
import subprocess
import asyncio
import shutil
from tqdm import tqdm
import lightgbm
import argparse


from util import *

def example_1():
    code = '''
class A:
    def __init__(self, value):
        self.value = value

    def demo_a(self, x):
        print('111111111111')
        print(x)
    
    def demo_b(self, x, y):
        self.demo_a(self.demo_a(y))
        self.demo_a(x)
        if outcome == 'odx85_cat' and SUFFIX != '_REV':
            print(1111)
        print(self.value)
        self.value = 22333

def demo_a(x):
    print('111111111111')
    print(x)
    '''
    code_function = '''def demo_b(self, x, y):
    self.demo_a(self.demo_a(y))
    self.demo_a(x)
    if outcome == 'odx85_cat' and SUFFIX != '_REV':
        print(1111)
    print(self.value)
    self.value = 22333
    '''
    code_function_type = 'class_method'
    return code, code_function, code_function_type

def example_2():
    code = '''
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np, pandas as pd
import json, ast
from matplotlib import pyplot as plt
from demo import xxx, yyy, zzz
import sklearn

class A:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        print('aaaa')

    def demo_a(self, x):
            
        print(x)

    def demo_aa(self, x):
        print('22222222')
        print(x)

    def demo_bb(self, x):
        print('333333333')
        print(x)

    def demo_b(self, x, y):
        if type(x) != pd.DataFrame:
            x = table(x)
        label_encoder = sklearn.preprocessing.LabelEncoder()
        var_a = var_b
        self.demo_a(x)
        self.demo_bb(y)

def demo_a(x):
    if SUFFIX != '_REV':
        print('No')
    print(x)

def demo_b(a, b):
    demo_a(a)
    demo_a(b)

def demo_c(x, y):
    tmp = 'aaaa'
    demo_b(x, y)
    q = np.mean([1,2,3])
    print(np.pi)
'''
    code_function = '''def demo_b(self, x, y):
    if type(x) != pd.DataFrame:
        x = table(x)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    var_a = var_b
    self.demo_a(x)
    self.demo_bb(y)
    '''
    code_function_type = 'class_method'
    return code, code_function, code_function_type

def walk_with_path(node, path=None):
    if path is None:
        path = []
    # Yield the current node and its path
    yield node, path
    # Recursively walk through the children nodes
    for child in ast.iter_child_nodes(node):
        yield from walk_with_path(child, path + [node])

def get_ast_context(ast_code_function, ast_code, original_ast_function_name):
    # code_function is a part of code
    # def complete_code_function():

    astCall_inCodeSnippet = []
    astAttribute_inCodeSnippet = []
    not_builtin_function_list = []
    not_builtin_attribute_list = []
    astFunctionDef_inCodeFile = {}
    astImport_inCodeFile = {}  # ast.Import & ast.ImportFrom
    astAssign_inCodeFile = {}
    ast_context = []

    for node in ast.walk(ast_code_function):
        if isinstance(node, ast.Call):
            # all the function call is from outside
            astCall_inCodeSnippet.append(node)
        elif isinstance(node, ast.Attribute):
            # all the function call is from outside
            astAttribute_inCodeSnippet.append(node)

    print('AST walking in ast_code_function.', flush=True)

    # solve the FunctionDef
    for astCall in astCall_inCodeSnippet:
        # only open for a name or attribute object
        if isinstance(astCall.func, ast.Name):
            astCall_name = astCall.func.id
            # judge whether the function is in-build or not?
            if astCall_name in dir(builtins):
                continue
            not_builtin_function_list.append([astCall_name, astCall])
        elif isinstance(astCall.func, ast.Attribute):
            # astCall.func.value might be ast.Call, here we ignore it
            if hasattr(astCall.func.value, 'id'):
                astCall_name = astCall.func.value.id + '.' + astCall.func.attr
                not_builtin_function_list.append([astCall_name, astCall])

    not_builtin_function_name_list = [x[0] for x in not_builtin_function_list]
    # print(not_builtin_function_name_list)
    for astAttribute in astAttribute_inCodeSnippet:
        if hasattr(astAttribute.value, 'id'):
            tmp_astAttribute_name = astAttribute.value.id + '.' + astAttribute.attr
            if tmp_astAttribute_name not in not_builtin_function_name_list:
                not_builtin_attribute_list.append([tmp_astAttribute_name, astAttribute])

    print('Collect not builtin functions.', flush=True)

    # for each not builtin function, we find the functionDef in side the whole code file
    for node, path in walk_with_path(ast_code):
        if isinstance(node, ast.FunctionDef):
            # all the function call is from outside
            node_prefix = ''
            for path_node in path:
                if isinstance(path_node, ast.ClassDef):
                    node_prefix += path_node.name + '.'

            if node_prefix and node.name == original_ast_function_name:
                for i in range(len(not_builtin_function_list)):
                    if 'self' in not_builtin_function_list[i][0]:
                        not_builtin_function_list[i][0] = not_builtin_function_list[i][0].replace('self.', node_prefix)


            astFunctionDef_inCodeFile[node_prefix + node.name] = node
        elif isinstance(node, ast.Import):
            for astAlias in node.names:
                new_node = copy.deepcopy(node)
                new_node.names = [astAlias]
                if astAlias.asname:
                    astImport_inCodeFile[astAlias.asname] = {'ast_node': new_node, 'name': astAlias.name}
                else:
                    astImport_inCodeFile[astAlias.name] = {'ast_node': new_node}
        elif isinstance(node, ast.ImportFrom):
            for astAlias in node.names:
                new_node = copy.deepcopy(node)
                new_node.names = [astAlias]
                if hasattr(astAlias, 'module'):
                    if astAlias.asname:
                        astImport_inCodeFile[astAlias.asname] = {'ast_node': new_node,
                                                                 'name': node.module + '.' + astAlias.name}
                    else:
                        astImport_inCodeFile[astAlias.name] = {'ast_node': new_node,
                                                               'name': node.module + '.' + astAlias.name}
                else:
                    if astAlias.asname:
                        astImport_inCodeFile[astAlias.asname] = {'ast_node': new_node,
                                                                 'name': astAlias.name}
                    else:
                        astImport_inCodeFile[astAlias.name] = {'ast_node': new_node}
        # elif isinstance(node, ast.Assign):

    for node in ast_code.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    astAssign_inCodeFile[target.id] = {'ast_node': node, 'name': target.id}


    print('AST walk in ast_code.', flush=True)

    # name matching
    for not_builtin_function in not_builtin_function_list:
        astCall_name = not_builtin_function[0]
        if astCall_name in astFunctionDef_inCodeFile:
            # ast_context.append(ast.unparse(astFunctionDef_inCodeFile[astCall_name][-1])
            if astFunctionDef_inCodeFile[astCall_name] not in ast_context:
                ast_context.append(astFunctionDef_inCodeFile[astCall_name])
        else:
            astCall = not_builtin_function[1]
            # tmp_astname = astCall_name.split('.')[0]
            if hasattr(astCall.func, 'id'):
                if astCall.func.id in astImport_inCodeFile:
                    if astImport_inCodeFile[astCall.func.id]['ast_node'] not in ast_context:
                        ast_context.append(astImport_inCodeFile[astCall.func.id]['ast_node'])
                elif astCall.func.id in astAssign_inCodeFile:
                    if astAssign_inCodeFile[astCall.func.id]['ast_node'] not in ast_context:
                        ast_context.append(astAssign_inCodeFile[astCall.func.id]['ast_node'])
            else:
                tmp_ast_node = astCall.func
                while hasattr(tmp_ast_node, 'value'):
                    if tmp_ast_node.value.id in astImport_inCodeFile:
                        if astImport_inCodeFile[tmp_ast_node.value.id]['ast_node'] not in ast_context:
                            ast_context.append(astImport_inCodeFile[tmp_ast_node.value.id]['ast_node'])
                    elif tmp_ast_node.value.id in astAssign_inCodeFile:
                        if astAssign_inCodeFile[tmp_ast_node.value.id]['ast_node'] not in ast_context:
                            ast_context.append(astAssign_inCodeFile[tmp_ast_node.value.id]['ast_node'])
                    tmp_ast_node = tmp_ast_node.value

    for not_builtin_attribute in not_builtin_attribute_list:
        astAttribute_name = not_builtin_attribute[0]
        if astAttribute_name in astFunctionDef_inCodeFile:
            if astFunctionDef_inCodeFile[astAttribute_name] not in ast_context:
                ast_context.append(astFunctionDef_inCodeFile[astAttribute_name])
        else:
            astAttribute = not_builtin_attribute[1]
            if astAttribute.value.id in astImport_inCodeFile:
                if astImport_inCodeFile[astAttribute.value.id]['ast_node'] not in ast_context:
                    ast_context.append(astImport_inCodeFile[astAttribute.value.id]['ast_node'])
            elif astAttribute.value.id in astAssign_inCodeFile:
                if astAssign_inCodeFile[astAttribute.value.id]['ast_node'] not in ast_context:
                    ast_context.append(astAssign_inCodeFile[astAttribute.value.id]['ast_node'])

    res_dic = {
        'astCall_inCodeSnippet': astCall_inCodeSnippet,
        'astAttribute_inCodeSnippet': astAttribute_inCodeSnippet,
        'not_builtin_function_list': not_builtin_function_list,
        'not_builtin_attribute_list': not_builtin_attribute_list,
        'astFunctionDef_inCodeFile': astFunctionDef_inCodeFile,
        'astImport_inCodeFile': astImport_inCodeFile,
        'astAssign_inCodeFile': astAssign_inCodeFile,
        'ast_context': ast_context
    }

    return res_dic

def ast_context_processing(ast_context, inCodeFile_dic):
    final_list = []
    # ast_context = list(set(ast_context))
    ast_context = sorted(ast_context, key=lambda x: x[1], reverse=True)
    # import first
    for ast_object, id in ast_context:
        if isinstance(ast_object, ast.Import) and ast_object not in final_list:
            final_list.append(remove_self(ast_object, inCodeFile_dic))
        elif isinstance(ast_object, ast.ImportFrom) and ast_object not in final_list:
            final_list.append(remove_self(ast_object, inCodeFile_dic))

    for ast_object, id in ast_context:
        if ast_object not in final_list:
            final_list.append(remove_self(ast_object, inCodeFile_dic))

    # for ast_object in final_list:
    #     remove_self(ast_object)

    return final_list

def combine_context_with_code(ast_context_id_list, ast_code, inCodeFile_dic):
    final_code = ast.unparse(ast_context_processing(ast_context_id_list, inCodeFile_dic))
    final_code += '\n\n# main code\n' + ast.unparse(ast_code)
    return final_code

def ast_node_remove_self(ast_code):
    class SelfAttributeRemover(ast.NodeTransformer):
        def visit_Attribute(self, node):
            # Check if the 'value' is an ast.Name with 'id' as 'self'
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                # Create a new AST node to replace the 'self' attribute
                # Here, let's change it to an `ast.Name` with the original attribute's name
                new_node = ast.Name(id=node.attr, ctx=ast.Load())
                return new_node

            # Otherwise, continue the normal visiting process
            return self.generic_visit(node)

    transformer = SelfAttributeRemover()
    modified_tree = transformer.visit(ast_code)
    return modified_tree

def modify_args_astFunctionDef(ast_node, inCodeFile_dic):
    class UndefinedVariableCollector(ast.NodeVisitor):
        def __init__(self, func_node):
            self.defined_vars = set()
            self.undefined_vars = set()
            self.func_node = func_node
            # Collect initial defined variables (function arguments), excluding 'self'
            for arg in func_node.args.args:
                if arg.arg != 'self':
                    self.defined_vars.add(arg.arg)

        def visit_Name(self, node):
            # If the node is a variable in Load context (used as a value)
            if isinstance(node.ctx, ast.Load):
                # filter out more args, such as library & their alias
                if (node.id not in self.defined_vars) and \
                        (not node.id.strip().startswith('*')) and \
                        (node.id not in dir(builtins)) and \
                        (node.id not in inCodeFile_dic['astImport_inCodeFile']) and \
                        (node.id not in inCodeFile_dic['astAssign_inCodeFile']) and \
                        (node.id not in inCodeFile_dic['astFunctionDef_inCodeFile']):
                    self.undefined_vars.add(node.id)
                    # print('QQQQQQQQQQQQ: %s' %  (node.id), flush=True)
            # If the node is being assigned to, add it to defined variables
            elif isinstance(node.ctx, ast.Store):
                self.defined_vars.add(node.id)
                # print('WWWWWWWWWWWWWWWWWWWW: %s' % (node.id), flush=True)
        def collect_undefined(self):
            # Visit all nodes in the function body
            self.visit(self.func_node)
            # Return the list of undefined variables
            return self.undefined_vars

    class AddUndefinedArgsTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Remove 'self' from the arguments, if present
            node.args.args = [arg for arg in node.args.args if arg.arg != 'self']

            # Collect undefined variables in the function
            collector = UndefinedVariableCollector(node)
            undefined_vars = collector.collect_undefined()

            # Create new arguments for undefined variables
            for var in undefined_vars:
                new_arg = ast.arg(arg=var, annotation=None)
                node.args.args.append(new_arg)

            return self.generic_visit(node)
            # return self.visit(node)

    transformer = AddUndefinedArgsTransformer()
    modified_tree = transformer.visit(ast_node)

    return modified_tree

    # add new args
    # args_name_list = []
    # for _arg in astFunctionDef.args.args:
    #     args_name_list.append(_arg.arg)
    #
    # for _attribute in not_builtin_attribute_list:
    #     if _attribute[1].attr not in args_name_list:
    #         new_arg = ast.arg(arg=_attribute[1].attr)
    #         astFunctionDef.args.args.append(new_arg)
    #         args_name_list.append(_attribute[1].attr)



def remove_self(ast_object, inCodeFile_dic):
    return modify_args_astFunctionDef(ast_node_remove_self(ast_object), inCodeFile_dic)

def get_ground_truth_code(code_function, code_function_type, code):
    # code, code_function, code_function_type = example_2()
    try:
        ast_code = ast.parse(code)
    except:
        print('Code file failed to be parsed into AST.')
        return None
    try:
        ast_code_function = ast.parse(code_function)
    except:
        print('Code function failed to be parsed into AST.')
        return None


    original_ast_function_name = ast_code_function.body[0].name
    ast_context_id_list = []

    res_dic = get_ast_context(ast_code_function, ast_code, original_ast_function_name)

    tmp_context = res_dic['ast_context']

    context_id = 0
    while tmp_context:
        print('Get the context...(%s)' % (context_id), flush=True)
        tmp_context_list = [x[0] for x in ast_context_id_list]
        for x in tmp_context:
            if x in tmp_context_list:
                tmp_context.remove(x)
            else:
                ast_context_id_list.append([x, context_id])
        refined_ast_context = ast.parse(ast.unparse(tmp_context))
        tmp_res_dic = get_ast_context(refined_ast_context, ast_code, original_ast_function_name)
        tmp_context = tmp_res_dic['ast_context']
        context_id += 1
        if context_id > 10:
            break

    # example 500, 0, 0 works
    # example 80, 0, 0 works
    inCodeFile_dic = {
        'astImport_inCodeFile': res_dic['astImport_inCodeFile'],
        'astAssign_inCodeFile': res_dic['astAssign_inCodeFile'],
        'astFunctionDef_inCodeFile': res_dic['astFunctionDef_inCodeFile']
    }
    if code_function_type == 'class_method':
        # ast_code_function = remove_self(ast_code_function, not_builtin_attribute_list)
        ast_code_function = remove_self(ast_code_function, inCodeFile_dic)
    ground_truth_code = combine_context_with_code(ast_context_id_list, ast_code_function, inCodeFile_dic)
    return ground_truth_code


async def run_script_pylint(target_file, i, total_length, destination_folder, semaphore):
    async with semaphore:
        try:
            print('%s out of %s' % (i, total_length), flush=True)
            process = await asyncio.create_subprocess_exec(
                "pylint", target_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=False,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            returncode = process.returncode

            res = {
                'target_file': target_file,
                'returncode': returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
            }

            pylint_result = res['stdout'].split('\n')
            pylint_check_flag = True
            for line in tqdm(pylint_result):
                tmp_list = line.split(':')
                if len(tmp_list) >= 4:
                    msg_id = tmp_list[3].strip()
                    if msg_id.startswith('E') or msg_id.startswith('F'):
                        pylint_check_flag = False
                        break
            if pylint_check_flag:
                shutil.copy(res['target_file'], destination_folder)
                # copy_tasks.append(async_copy_file_pylint(res['target_file'], destination_folder, semaphore))

        except asyncio.TimeoutError:
            pass
            # res = {
            #     'target_file': target_file,
            #     'returncode': -1,
            #     'stdout': '',
            #     'stderr': 'Process timed out'
            # }
        return
        # return res

async def async_copy_file_pylint(target_file, destination_folder, semaphore):
    async with semaphore:
        await asyncio.to_thread(shutil.copy, target_file, destination_folder)

async def compile_ground_truth_code_pylint(target_file_list, destination_folder, max_concurrent_copies=5):
    semaphore = asyncio.Semaphore(max_concurrent_copies)
    total_length = len(target_file_list)
    tasks = [run_script_pylint(file, i, total_length, destination_folder, semaphore) for i, file in enumerate(target_file_list)]
    await asyncio.gather(*tasks)

    # copy_tasks = []
    # for res in tqdm(res_list):
    #     pylint_result = res['stdout'].split('\n')
    #     pylint_check_flag = True
    #     for line in tqdm(pylint_result):
    #         tmp_list = line.split(':')
    #         if len(tmp_list) >= 4:
    #             msg_id = tmp_list[3].strip()
    #             if msg_id.startswith('E') or msg_id.startswith('F'):
    #                 pylint_check_flag = False
    #                 break
    #     if pylint_check_flag:
    #         copy_tasks.append(async_copy_file_pylint(res['target_file'], destination_folder, semaphore))
    #
    # await asyncio.gather(*copy_tasks)



# run this first
def generate_ground_truth_code(ds1000orstackoverflow='ds1000'):
    if ds1000orstackoverflow == 'ds1000':
        base_folder = 'intermediate_search_result/DS1000_search/'
        if not os.path.exists(base_folder + 'ground_truth_code/'):
            os.makedirs(base_folder + 'ground_truth_code/')
        for ds1000_id in range(1000):
            print('=========%s=========' % ds1000_id)
            file_path = os.path.join(base_folder + 'github_search_result_filtered', '%s.json' % (ds1000_id))

            with open(file_path, 'r') as f:
                res = json.load(f)

            for i in range(len(res['filtered_similar_code'])):
                code = res['filtered_similar_code'][i]['code']
                for j in range(len(res['filtered_similar_code'][i]['code_snippet_list'])):
                    print(ds1000_id, i, j)
                    save_file_path = base_folder + 'ground_truth_code/%s_%s_%s.py' % (ds1000_id, i, j)
                    if not os.path.exists(save_file_path):
                        code_function, code_function_type = res['filtered_similar_code'][i]['code_snippet_list'][j]
                        ground_truth_code = get_ground_truth_code(code_function, code_function_type, code)
                        if ground_truth_code:
                            with open(save_file_path, 'w') as f:
                                f.write(ground_truth_code)
    else:
        base_folder = 'intermediate_search_result/stackoverflow_search/'
        target_folder = base_folder + 'github_search_result_filtered/'
        file_list = os.listdir(target_folder)
        if not os.path.exists(base_folder + 'ground_truth_code/'):
            os.makedirs(base_folder + 'ground_truth_code/')
        for file_name in file_list:
            file_path = os.path.join(target_folder, file_name)
            try:
                with open(file_path, 'r') as f:
                    res = json.load(f)
            except:
                continue

            for i in range(len(res['filtered_similar_code'])):
                code = res['filtered_similar_code'][i]['code']
                for j in range(len(res['filtered_similar_code'][i]['code_snippet_list'])):
                    print(file_name, i, j)
                    save_file_path = base_folder + 'ground_truth_code/%s_%s_%s.py' % (file_name.replace('.json', ''), i, j)
                    if not os.path.exists(save_file_path):
                        code_function, code_function_type = res['filtered_similar_code'][i]['code_snippet_list'][j]
                        ground_truth_code = get_ground_truth_code(code_function, code_function_type, code)
                        if ground_truth_code:
                            with open(save_file_path, 'w') as f:
                                f.write(ground_truth_code)


# run this second
def compilation_filter(ds1000orstackoverflow='ds1000', library='keras'):
    if ds1000orstackoverflow == 'ds1000':
        base_folder = 'intermediate_search_result/DS1000_search/'
    else:
        base_folder = 'intermediate_search_result/stackoverflow_search/'
    destination_folder = base_folder + 'ground_truth_code_filtered/'
    folder_path = base_folder + 'ground_truth_code/'

    target_file_list = []
    for file_name in tqdm(os.listdir(folder_path)):
        target_file = os.path.join(folder_path, file_name)
        if library not in file_name:
            continue
        if os.path.exists(target_file.replace('ground_truth_code', 'ground_truth_code_filtered')):
            continue
        target_file_list.append(target_file)

    print(target_file_list)
    print(destination_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    asyncio.run(compile_ground_truth_code_pylint(target_file_list, destination_folder, 10))


# run this third
def deduplication(ds1000orstackoverflow='ds1000'):
    if ds1000orstackoverflow == 'ds1000':
        base_folder = 'intermediate_search_result/DS1000_search/'
        tmp_dic = {}
        folder_path = base_folder + 'ground_truth_code_filtered/'
        for file_name in sorted(os.listdir(folder_path), key=lambda x: int(x.split('/')[-1].split('_')[0])):
            target_file = os.path.join(folder_path, file_name)
            with open(target_file, 'r') as f:
                code = f.read()
            if code not in tmp_dic:
                tmp_dic[code] = target_file
    else:
        base_folder = 'intermediate_search_result/stackoverflow_search/'
        tmp_dic = {}
        folder_path = base_folder + 'ground_truth_code_filtered/'
        ds1000_folder_path = 'intermediate_search_result/DS1000_search/' + 'ground_truth_code_filtered/'
        for file_name in sorted(os.listdir(ds1000_folder_path), key=lambda x: int(x.split('/')[-1].split('_')[0])):
            target_file = os.path.join(ds1000_folder_path, file_name)
            with open(target_file, 'r') as f:
                code = f.read()
            if code not in tmp_dic:
                tmp_dic[code] = target_file

        for file_name in sorted(os.listdir(folder_path), key=lambda x: int(x.split('/')[-1].split('_')[1])):
            target_file = os.path.join(folder_path, file_name)
            with open(target_file, 'r') as f:
                code = f.read()
            if code not in tmp_dic:
                tmp_dic[code] = target_file

        for file_name in sorted(os.listdir(ds1000_folder_path), key=lambda x: int(x.split('/')[-1].split('_')[0])):
            target_file = os.path.join(ds1000_folder_path, file_name)
            with open(target_file, 'r') as f:
                code = f.read()
            if code in tmp_dic:
                tmp_dic.pop(code)

    destination_folder = base_folder + 'ground_truth_code_filtered_deduplicated/'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for key in tmp_dic:
        shutil.copy(tmp_dic[key], destination_folder)


# Pandas 0-290
# Numpy 291-510
# Matplotlib 511-665
# Tensorflow 666-710
# Scipy 711-816
# Sklearn 817-931
# Pytorch 932-999


#### result analysis=============================================================
def load_ground_truth_dic():
    folder_path = 'ground_truth_code_filtered_deduplicated/'
    # folder_path = 'ground_truth_code/'

    ground_truth_code_dic = {
        'pandas': [],
        'numpy': [],
        'matplotlib': [],
        'tensorflow': [],
        'scipy': [],
        'sklearn': [],
        'pytorch': []
    }
    for file_name in os.listdir(folder_path):
        target_file = os.path.join(folder_path, file_name)
        # print(target_file)
        try:
            if int(file_name.split('_')[0]) <= 290:
                ground_truth_code_dic['pandas'].append(target_file)
            elif int(file_name.split('_')[0]) <= 510:
                ground_truth_code_dic['numpy'].append(target_file)
            elif int(file_name.split('_')[0]) <= 665:
                ground_truth_code_dic['matplotlib'].append(target_file)
            elif int(file_name.split('_')[0]) <= 710:
                ground_truth_code_dic['tensorflow'].append(target_file)
            elif int(file_name.split('_')[0]) <= 816:
                ground_truth_code_dic['scipy'].append(target_file)
            elif int(file_name.split('_')[0]) <= 931:
                ground_truth_code_dic['sklearn'].append(target_file)
            else:
                ground_truth_code_dic['pytorch'].append(target_file)
        except:
            continue
        # ground_truth_code_list.append(target_file)
    return ground_truth_code_dic

def ground_truth_statistics(option='pandas'):
    ground_truth_code_dic = load_ground_truth_dic()
    code_line = 0
    code_word = 0
    file_count = 0
    if option == 'all':
        for library_name in ground_truth_code_dic:
            for file_name in ground_truth_code_dic[library_name]:
                with open(file_name, 'r') as f:
                    for line in f:
                        code_line += 1
                        words = line.split()
                        code_word += len(words)
                file_count += 1
    else:
        for file_name in ground_truth_code_dic[option]:
            with open(file_name, 'r') as f:
                for line in f:
                    code_line += 1
                    words = line.split()
                    code_word += len(words)
            file_count += 1
    print('%s\t%s\t%s\t%s' % (option, file_count, round(code_line/file_count, 2), round(code_word/file_count, 2)))

def makeup_tensorflow():
    folder_path = 'ground_truth_code/'
    tensorflow_list = []

    for file_name in os.listdir(folder_path):
        target_file = os.path.join(folder_path, file_name)
        # print(target_file)
        if int(file_name.split('_')[0]) > 816 and int(file_name.split('_')[0]) <= 931:
            tensorflow_list.append(target_file)

    folder_path = 'ground_truth_code_filtered/'
    tmp_filtered_list = []
    for file_name in os.listdir(folder_path):
        target_file = os.path.join(folder_path, file_name)
        # print(target_file)
        if int(file_name.split('_')[0]) > 816 and int(file_name.split('_')[0]) <= 931:
            tmp_filtered_list.append(file_name)

    unselected_tmp_list = []
    for x in tensorflow_list:
        if x.split('/')[-1] not in tmp_filtered_list:
            unselected_tmp_list.append(x)

    unselected_tensorflow_list = sorted(unselected_tmp_list)
    a = load_ground_truth_dic()
    # for file_name in unselected_tensorflow_list:
    #     # file_name = unselected_tensorflow_list[3]
    #     print(file_name)
    #     with open(file_name, 'r') as f:
    #         code = f.read()
    #     with open('filter_code_demo_2.py', 'w') as f:
    #         f.write(code)
    #     output = subprocess.run(["python", 'filter_code_demo_2.py'], capture_output=True, text=True)
    #     if output.returncode == 0:
    #         with open('filter_code_demo_2.py', 'r') as f:
    #             new_code = f.read()
    #         with open(file_name.replace('ground_truth_code', 'ground_truth_code_filtered'), 'w') as f:
    #             f.write(new_code)

def demo_script():
    store_file_path = 'ground_truth_manual_filter.txt'
    ground_truth_manual_filter_dic = {}
    tmp_list = os.listdir('ground_truth_code_filtered_deduplicated')
    if os.path.exists(store_file_path):
        with open(store_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # count += 1
                # print(count)
                content = json.loads(line)
                if content[0].split('/')[-1] in tmp_list:
                    if len(content) == 2:
                        ground_truth_manual_filter_dic[content[0].replace('ground_truth_code_filtered', 'ground_truth_code_filtered_deduplicated')] = [content[1]]
                    else:
                        ground_truth_manual_filter_dic[content[0].replace('ground_truth_code_filtered', 'ground_truth_code_filtered_deduplicated')] = [content[1], content[2]]
            # ground_truth_manual_filter_dic = json.load(f)
    else:
        ground_truth_manual_filter_dic = {}

    for key in ground_truth_manual_filter_dic:
        with open('ground_truth_manual_filter_new.txt', 'a') as f:
            if len(ground_truth_manual_filter_dic[key]) == 1:
                f.write(json.dumps([key, ground_truth_manual_filter_dic[key][0]]) + '\n')
            else:
                f.write(json.dumps(
                    [key, ground_truth_manual_filter_dic[key][0], ground_truth_manual_filter_dic[key][1]]) + '\n')


# pandas
def manual_filter():
    # filter the ground truth by hand
    count = 0
    store_file_path = 'ground_truth_manual_filter.txt'
    ground_truth_manual_filter_dic = {}
    if os.path.exists(store_file_path):
        with open(store_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                count += 1
                print(count)
                content = json.loads(line)
                ground_truth_manual_filter_dic[content[0]] = content[1]
            # ground_truth_manual_filter_dic = json.load(f)
    else:
        ground_truth_manual_filter_dic = {}

    folder_path = 'ground_truth_code_filtered_deduplicated/'

    for file_name in sorted(os.listdir(folder_path), key=lambda x: int(x.split('/')[-1].split('_')[0])):
        target_file = os.path.join(folder_path, file_name)
        print(target_file)
        if target_file not in ground_truth_manual_filter_dic:
            with open(target_file, 'r') as f:
                code = f.read()

            with open('filter_code_demo.py', 'w') as f:
                f.write(code)

            a_input = input()
            with open('filter_code_demo.py', 'r') as f:
                new_code = f.read()
            if a_input == '1':
                tmp_var = [target_file, True, new_code]
            else:
                tmp_var = [target_file, False]

            with open(store_file_path, 'a') as f:
                f.write(json.dumps(tmp_var) + '\n')

# ================================================================================

## example trail code
def example():
    # code, code_function, code_function_type = example_1()
    # ds1000_id, i, j = 291, 10, 3
    ds1000_id, i, j = 919, 2, 1
    file_path = os.path.join('github_search_result_filtered', '%s.json' % (ds1000_id))
    with open(file_path, 'r') as f:
            res = json.load(f)
    code = res['filtered_similar_code'][i]['code']
    code_function, code_function_type = res['filtered_similar_code'][i]['code_snippet_list'][j]
    try:
        ast_code = ast.parse(code)
    except:
        print('Code file failed to be parsed into AST.')

    try:
        ast_code_function = ast.parse(code_function)
    except:
        print('Code function failed to be parsed into AST.')

    original_ast_function_name = ast_code_function.body[0].name
    ast_context_id_list = []

    res_dic = get_ast_context(ast_code_function, ast_code, original_ast_function_name)

    tmp_context = res_dic['ast_context']

    context_id = 0
    while tmp_context:
        print('Get the context...(%s)' % (context_id), flush=True)
        tmp_context_list = [x[0] for x in ast_context_id_list]
        for x in tmp_context:
            if x in tmp_context_list:
                tmp_context.remove(x)
            else:
                ast_context_id_list.append([x, context_id])
        refined_ast_context = ast.parse(ast.unparse(tmp_context))
        tmp_res_dic = get_ast_context(refined_ast_context, ast_code, original_ast_function_name)
        tmp_context = tmp_res_dic['ast_context']
        context_id += 1
        if context_id > 10:
            break

    inCodeFile_dic = {
        'astImport_inCodeFile': res_dic['astImport_inCodeFile'],
        'astAssign_inCodeFile': res_dic['astAssign_inCodeFile'],
        'astFunctionDef_inCodeFile': res_dic['astFunctionDef_inCodeFile']
    }

    # if code_function_type == 'class_method':
        # ast_code_function = remove_self(ast_code_function, not_builtin_attribute_list)
    ast_code_function = remove_self(ast_code_function, inCodeFile_dic)
    ground_truth_code = combine_context_with_code(ast_context_id_list, ast_code_function, inCodeFile_dic)


    with open('filter_code_demo_2.py', 'w') as f:
        f.write(ground_truth_code)


# generate_ground_truth_code('s')
deduplication('s')
# if __name__ == '__main__':
#     # compilation_filter('stackoverflow', 'keras')
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-l",
#         "--library",
#         type=str,
#         choices=['numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn',
#                  'tensorflow', 'pytorch', 'seaborn', 'keras', 'lightgbm', 'all'],
#         help="Choose library",
#         required=True,
#     )
#     args = parser.parse_args()
#     compilation_filter('stackoverflow', args.library)



# base_folder = 'intermediate_search_result/stackoverflow_search/'
# destination_folder = base_folder + 'ground_truth_code_filtered/'
# folder_path = base_folder + 'ground_truth_code/'
# library = 'keras'
# target_file_list = []
# for file_name in tqdm(os.listdir(folder_path)):
#     target_file = os.path.join(folder_path, file_name)
#     if library not in file_name:
#         continue
#     if os.path.exists(target_file.replace('ground_truth_code', 'ground_truth_code_filtered')):
#         continue
#     target_file_list.append(target_file)