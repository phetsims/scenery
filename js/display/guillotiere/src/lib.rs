use guillotiere::*;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// See generate.js for generation

#[wasm_bindgen]
pub struct GuillotiereBin {
    allocation: Option<Allocation>,
}

#[wasm_bindgen]
impl GuillotiereBin {
    pub fn is_valid(&self) -> bool {
        self.allocation.is_some()
    }

    #[wasm_bindgen(method, getter = id)]
    pub fn get_id(&self) -> u32 {
        self.allocation.unwrap().id.serialize()
    }

    #[wasm_bindgen(method, getter = x)]
    pub fn get_x(&self) -> i32 {
        self.allocation.unwrap().rectangle.min.x
    }

    #[wasm_bindgen(method, getter = y)]
    pub fn get_y(&self) -> i32 {
        self.allocation.unwrap().rectangle.min.y
    }
}

#[wasm_bindgen]
pub struct GuillotiereAtlas {
    allocator: AtlasAllocator,
}

#[wasm_bindgen]
impl GuillotiereAtlas {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32) -> GuillotiereAtlas {
        GuillotiereAtlas {
            allocator: AtlasAllocator::new(size2(width, height)),
        }
    }

    pub fn allocate(&mut self, width: i32, height: i32) -> GuillotiereBin {
        GuillotiereBin {
            allocation: self.allocator.allocate(size2(width, height)),
        }
    }

    pub fn deallocate(&mut self, id: u32) {
        self.allocator.deallocate(AllocId::deserialize(id));
    }

    pub fn is_empty(&self) -> bool {
        self.allocator.is_empty()
    }

    pub fn clear(&mut self) {
        self.allocator.clear();
    }

    pub fn rearrange(&mut self) -> String {
        let change_list = self.allocator.rearrange().clone();
        self.process_change_list(change_list)
    }

    pub fn resize_and_rearrange(&mut self, width: i32, height: i32) -> String {
        let change_list = self.allocator.resize_and_rearrange(size2(width, height));
        self.process_change_list(change_list)
    }

    pub fn grow(&mut self, width: i32, height: i32) {
        self.allocator.grow(size2(width, height));
    }

    // NOTE: It was a struggle to get this out in a more "normal" way. JSON parsing is suboptimal
    // IDEALLY we could handle the "remapping" on the rust end here
    fn process_change_list(&self, change_list: ChangeList) -> String {
        let mut result = String::new();

        result.push_str("{ \"changes\": [ ");

        let mut is_first = true;
        for change in change_list.changes {
            let old_allocation = change.old;
            let new_allocation = change.new;

            if (!is_first) {
                result.push_str(",");
            }
            is_first = false;

            result.push_str(&format!(
                "{{ \"old_id\": {}, \"new_id\": {}, \"new_x\": {}, \"new_y\": {} }}",
                old_allocation.id.serialize(),
                new_allocation.id.serialize(),
                new_allocation.rectangle.min.x,
                new_allocation.rectangle.min.y
            ));
        }

        result.push_str(" ], \"failures\": [ ");

        let mut is_first = true;
        for failure in change_list.failures {
            let old_allocation = failure;

            if (!is_first) {
                result.push_str(",");
            }
            is_first = false;

            result.push_str(&format!(
                "{{ \"old_id\": {} }}",
                old_allocation.id.serialize()
            ));
        }

        result.push_str(" ] }");

        result
    }
}

#[wasm_bindgen(start)]
fn run() {}
