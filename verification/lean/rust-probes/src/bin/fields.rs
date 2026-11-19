//! Extract simple, explicitly written primitive getters and the independently declared field index.
use std::collections::HashMap;
use std::io::{self, BufRead};
use syn::visit::{self, Visit};
use syn::{Expr, Fields, ImplItem, ItemImpl, ItemStruct, ReturnType, Stmt, Type};

#[derive(Default)]
struct Declarations<'a> {
    structs: HashMap<String, Vec<&'a Fields>>,
    impls: Vec<&'a ItemImpl>,
}

impl<'a> Visit<'a> for Declarations<'a> {
    fn visit_item_struct(&mut self, item: &'a ItemStruct) {
        self.structs
            .entry(item.ident.to_string())
            .or_default()
            .push(&item.fields);
        visit::visit_item_struct(self, item);
    }
    fn visit_item_impl(&mut self, item: &'a ItemImpl) {
        if item.trait_.is_none() {
            self.impls.push(item);
        }
        visit::visit_item_impl(self, item);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args().nth(1).ok_or("Pass the repository root")?;
    println!("file\timpl_line\timpl_col\tmethod\ttype\tfield\tsource_index\twidth");
    for file in io::stdin().lock().lines() {
        let file = file?;
        let src = std::fs::read_to_string(std::path::Path::new(&root).join(&file))?;
        let ast = syn::parse_file(&src)?;
        let mut declarations = Declarations::default();
        declarations.visit_file(&ast);
        for item in declarations.impls {
            let Type::Path(owner) = item.self_ty.as_ref() else {
                continue;
            };
            let Some(owner) = owner.path.segments.last() else {
                continue;
            };
            let Some([fields]) = declarations
                .structs
                .get(&owner.ident.to_string())
                .map(Vec::as_slice)
            else {
                continue;
            };
            for method in &item.items {
                let ImplItem::Fn(method) = method else {
                    continue;
                };
                if method.sig.inputs.len() != 1 || method.sig.receiver().is_none() {
                    continue;
                }
                let ReturnType::Type(_, ty) = &method.sig.output else {
                    continue;
                };
                let Type::Path(ty) = ty.as_ref() else {
                    continue;
                };
                let Some(ty) = ty.path.get_ident() else {
                    continue;
                };
                let ty = ty.to_string();
                if !matches!(
                    ty.as_str(),
                    "bool"
                        | "f32"
                        | "f64"
                        | "i8"
                        | "i16"
                        | "i32"
                        | "i64"
                        | "i128"
                        | "isize"
                        | "u8"
                        | "u16"
                        | "u32"
                        | "u64"
                        | "u128"
                        | "usize"
                ) {
                    continue;
                }
                let [Stmt::Expr(Expr::Field(read), None)] = method.block.stmts.as_slice() else {
                    continue;
                };
                let Expr::Path(base) = read.base.as_ref() else {
                    continue;
                };
                if !base.path.is_ident("self") {
                    continue;
                }
                let (field, index) = match &read.member {
                    syn::Member::Named(name) => {
                        let index = fields
                            .iter()
                            .position(|f| f.ident.as_ref() == Some(name))
                            .ok_or_else(|| format!("Missing declared field: {file} {name}"))?;
                        (name.to_string(), index)
                    }
                    syn::Member::Unnamed(n) => (n.index.to_string(), n.index as usize),
                };
                let start = item.impl_token.span.start();
                println!(
                    "{file}\t{}\t{}\t{}\t{ty}\t{field}\t{index}\t{}",
                    start.line,
                    start.column + 1,
                    method.sig.ident,
                    fields.len()
                );
            }
        }
    }
    Ok(())
}
