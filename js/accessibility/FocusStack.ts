// Copyright 2025, University of Colorado Boulder

/**
 * A stack structure for managing focus. It provides operations to push, pop, peek, and clear focusable nodes,
 * as well as utility functions to check the stack size and emptiness.
 *
 * StackElement can be either a Node or a callback function that returns a Node. The callback form is useful
 * when the Node to be focused should be determined dynamically at the time of focus, rather than when it is
 * pushed onto the stack. This allows for deferred creation or selection of the focusable Node.
 *
 * Example:
 *
 * const focusStack = new FocusStack();
 * focusStack.push(node);
 *
 * const topNode = focusStack.peek();
 * focusStack.popFocus(); // Pops and focuses the top Node
 *
 * @author Jesse Greenberg
 */

import Node from '../nodes/Node.js';
import scenery from '../scenery.js';

type StackElement = Node | ( () => Node );

class FocusStack {
  private readonly stack: StackElement[];

  /**
   * Constructs a new FocusStack instance.
   * Initializes an empty stack to store focusable nodes.
   */
  public constructor() {
    this.stack = [];
  }

  /**
   * Pushes a focusable node onto the stack.
   */
  public push( element: StackElement ): void {
    this.stack.push( element );
  }

  /**
   * Pops the top focusable node from the stack and returns it.
   */
  public pop(): StackElement | undefined {
    return this.stack.pop();
  }

  /**
   * Peeks at the top focusable node without removing it from the stack.
   */
  public peek(): StackElement | undefined {
    return this.stack[ this.stack.length - 1 ];
  }

  /**
   * Pops the top focusable node from the stack and gives it focus.
   */
  public popFocus(): Node | undefined {
    const nextItem = this.pop();

    if ( typeof nextItem === 'function' ) {
      const node = nextItem();
      node?.focus();
      return node;
    }
    else {
      nextItem?.focus();
      return nextItem;
    }
  }

  /**
   * Clears all nodes from the focus stack.
   */
  public clear(): void {
    this.stack.length = 0;
  }

  /**
   * Gets the current number of nodes in the focus stack.
   */
  public size(): number {
    return this.stack.length;
  }

  /**
   * Checks if the focus stack is empty.
   */
  public isEmpty(): boolean {
    return this.stack.length === 0;
  }
}

scenery.register( 'FocusStack', FocusStack );

export default FocusStack;