//! Inventory explicitly declared functions before macro expansion, including inline test modules.
use std::io::{self, BufRead, Write};
use syn::visit::{self, Visit};

struct Inventory<'a, W> {
    file: &'a str,
    modules: Vec<String>,
    writer: W,
}

impl<W: Write> Inventory<'_, W> {
    fn record(&mut self, kind: &str, sig: &syn::Signature, attr: &[syn::Attribute]) {
        let is_test = attr.iter().any(|a| a.path().is_ident("test"));
        writeln!(
            self.writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.file,
            sig.ident.span().start().line,
            self.modules.join("::"),
            kind,
            sig.ident,
            is_test
        )
        .unwrap();
    }
}

impl<'a, W: Write> Visit<'a> for Inventory<'_, W> {
    fn visit_item_mod(&mut self, item: &'a syn::ItemMod) {
        self.modules.push(item.ident.to_string());
        visit::visit_item_mod(self, item);
        self.modules.pop();
    }
    fn visit_item_fn(&mut self, item: &'a syn::ItemFn) {
        self.record("function", &item.sig, &item.attrs);
        visit::visit_item_fn(self, item);
    }
    fn visit_impl_item_fn(&mut self, item: &'a syn::ImplItemFn) {
        self.record("method", &item.sig, &item.attrs);
        visit::visit_impl_item_fn(self, item);
    }
    fn visit_trait_item_fn(&mut self, item: &'a syn::TraitItemFn) {
        self.record("trait", &item.sig, &item.attrs);
        visit::visit_trait_item_fn(self, item);
    }
    fn visit_foreign_item_fn(&mut self, item: &'a syn::ForeignItemFn) {
        self.record("foreign", &item.sig, &item.attrs);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::args().nth(1).ok_or("Pass the repository root")?;
    let mut writer = io::BufWriter::new(io::stdout().lock());
    writeln!(
        writer,
        "file\tline\tinline_module\tkind\tname\texplicit_test"
    )?;
    for file in io::stdin().lock().lines() {
        let file = file?;
        let text = std::fs::read_to_string(std::path::Path::new(&root).join(&file))?;
        let ast = syn::parse_file(&text).map_err(|error| format!("{file}: {error}"))?;
        Inventory {
            file: &file,
            modules: Vec::new(),
            writer: &mut writer,
        }
        .visit_file(&ast);
    }
    Ok(())
}
