// Copyright 2021, University of Colorado Boulder

import TinyProperty from '../../../axon/js/TinyProperty.js';
declare type Constructor<T = {}> = new ( ...args: any[] ) => T;
declare const HeightSizable: <SuperType extends Constructor>( key: SuperType, ...args: any[] ) => {
    new ( ...args: any[] ): {
        preferredHeightProperty: TinyProperty;
        minimumHeightProperty: TinyProperty;
        heightSizable: true;
        preferredHeight: number | null;
        minimumHeight: number | null;
    };
} & SuperType;
export default HeightSizable;