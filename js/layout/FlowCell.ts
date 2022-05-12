// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for FlowConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { FlowConfigurable, FlowConfigurableOptions, FlowConstraint, LayoutAlign, LayoutProxy, MarginLayoutCell, Node, scenery } from '../imports.js';

export default class FlowCell extends FlowConfigurable( MarginLayoutCell ) {

  _pendingSize: number; // scenery-internal

  flowConstraint: FlowConstraint;

  constructor( constraint: FlowConstraint, node: Node, proxy: LayoutProxy | null ) {
    super( constraint, node, proxy );

    this.flowConstraint = constraint;
    this._pendingSize = 0;

    this.orientation = constraint.orientation;
    this.onLayoutOptionsChange();
  }

  get effectiveAlign(): LayoutAlign {
    return this._align !== null ? this._align : this.flowConstraint._align!;
  }

  get effectiveStretch(): boolean {
    return this._stretch !== null ? this._stretch : this.flowConstraint._stretch!;
  }

  get effectiveGrow(): number {
    return this._grow !== null ? this._grow : this.flowConstraint._grow!;
  }

  protected override onLayoutOptionsChange(): void {
    if ( this.node.layoutOptions ) {
      this.setOptions( this.node.layoutOptions as FlowConfigurableOptions );
    }

    super.onLayoutOptionsChange();
  }

  private setOptions( options?: FlowConfigurableOptions ): void {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }
}

scenery.register( 'FlowCell', FlowCell );
