// Copyright 2022, University of Colorado Boulder

/**
 * A Node represented by a heading in the parallel dom that can receive focus. Typically
 * headings are not focusable and not interactive. But it may be desirable to put focus
 * on a heading to orient the user or control where the traversal order starts without
 * focusing an interactive component.
 *
 * When a screen reader is focused on a heading it will read the name of the heading and
 * possibly the content below it.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import optionize from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { Node, NodeOptions, scenery } from '../../imports.js';

// Available heading levels, according to DOM spec.
type HeadingLevelNumber = 1 | 2 | 3 | 4 | 5 | 6;

type SelfOptions = {

  // The heading level for this focusable heading in the PDOM, 1-6 according to DOM spec.
  headingLevel?: HeadingLevelNumber;
};
type ParentOptions = StrictOmit<NodeOptions, 'tagName' | 'focusHighlight'>;
export type FocusableHeadingNodeOptions = SelfOptions & ParentOptions;

class FocusableHeadingNode extends Node {

  // Removes listeners and makes eligible for garbage collection.
  private readonly disposeFocusableHeadingNode: () => void;

  public constructor( providedOptions?: FocusableHeadingNodeOptions ) {
    const options = optionize<FocusableHeadingNodeOptions, SelfOptions, ParentOptions>()( {
      headingLevel: 1
    }, providedOptions );

    super( options );

    this.tagName = `h${options.headingLevel}`;

    // This Node is focusable but there is no interactive component to surround with a highlight.
    this.focusHighlight = 'invisible';

    // After losing focus, this element is removed from the traversal order. It can only receive
    // focus again after calling focus() directly.
    const blurListener = {
      blur: () => { this.focusable = false; }
    };
    this.addInputListener( blurListener );

    this.disposeFocusableHeadingNode = () => {
      this.removeInputListener( blurListener );
    };
  }

  /**
   * Focus this heading in the Parallel DOM. The screen reader will read its name and possibly
   * content below it. Traversal with alternative input will continue from wherever this element
   * is located in the PDOM order.
   *
   * Once the heading loses focus, it is removed from the traversal order until this is called
   * explicitly again.
   */
  public override focus(): void {
    this.focusable = true;
    super.focus();
  }

  public override dispose(): void {
    this.disposeFocusableHeadingNode();
    super.dispose();
  }
}

scenery.register( 'FocusableHeadingNode', FocusableHeadingNode );
export default FocusableHeadingNode;
